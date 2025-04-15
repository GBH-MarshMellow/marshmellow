// lib/presentation/viewmodels/encryption/encryption_viewmodel.dart

import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:dio/dio.dart';
import 'package:encrypt/encrypt.dart' as encrypt;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:marshmellow/core/config/app_config.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/data/datasources/remote/encryption/encrypt_api.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'package:marshmellow/di/providers/core_providers.dart';
import 'package:pointycastle/api.dart';
import 'package:pointycastle/asymmetric/api.dart';
import 'package:pointycastle/asymmetric/pkcs1.dart';
import 'package:pointycastle/asymmetric/rsa.dart';

// 암호화 API 프로바이더
final encryptApiProvider = Provider((ref) {
  final apiClient = ref.read(apiClientProvider);
  return EncryptApi(apiClient);
});

class EncryptionService {
  final FlutterSecureStorage _secureStorage;
  
  EncryptionService(this._secureStorage);
  
  // RSA 개인키로 암호화된 AES 키 복호화 (PKCS1 패딩 사용)
  Future<String> decryptAesKey(String encryptedAesKey) async {
    try {
      // Base64로 디코딩
      final encryptedBytes = base64Decode(encryptedAesKey);
      
      // RSA 개인키 파싱
      final parser = encrypt.RSAKeyParser();
      final privateKey = parser.parse(AppConfig.rsaPrivateKey) as RSAPrivateKey;
      
      // PKCS1 패딩으로 복호화
      final decrypter = PKCS1Encoding(RSAEngine())
        ..init(false, PrivateKeyParameter<RSAPrivateKey>(privateKey));
      
      final output = Uint8List(decrypter.outputBlockSize);
      final outLen = decrypter.processBlock(encryptedBytes, 0, encryptedBytes.length, output, 0);
      final decryptedBytes = output.sublist(0, outLen);
      
      // 복호화된 AES 키를 base64로 인코딩해서 반환
      final decryptedBase64 = base64Encode(decryptedBytes);
      return decryptedBase64;
    } catch (e) {
      print('AES 키 복호화 실패: $e');
      rethrow;
    }
  }
  
  // AES 키 저장 (base64 인코딩된 형태로 저장)
  Future<void> saveAesKey(String aesKeyBase64) async {
    await _secureStorage.write(key: StorageKeys.aesKey, value: aesKeyBase64);
  }
  
  // AES 키 조회 (base64 인코딩된 형태로 반환)
  Future<String?> getAesKey() async {
    return await _secureStorage.read(key: StorageKeys.aesKey);
  }

  // Future<String?> getAesKey() async {
  //   final keyByteee = await _secureStorage.read(key: StorageKeys.certificatePem);
  //   print('@@@@@@@@@@@@dd@@@@@@');
    
  //   // 긴 문자열을 여러 부분으로 나누어 출력
  //   if (keyByteee != null) {
  //     const int chunkSize = 500; // 한 번에 출력할 문자 수
  //     for (int i = 0; i < keyByteee.length; i += chunkSize) {
  //       int end = (i + chunkSize < keyByteee.length) ? i + chunkSize : keyByteee.length;
  //       print('Part ${i ~/ chunkSize + 1}: ${keyByteee.substring(i, end)}');
  //     }
  //   } else {
  //     print('keyByteee is null');
  //   }
    
  //   return keyByteee;
  // }

  // AES 암호화 메서드 (향후 사용)
  Future<String> encryptWithAes(String plainText) async {
    final aesKeyBase64 = await getAesKey();
    if (aesKeyBase64 == null) {
      throw Exception('AES 키가 없습니다');
    }
    
    // base64 디코딩해서 바이트 배열로 변환
    final keyBytes = base64Decode(aesKeyBase64);
    
    // encrypt 패키지용 키로 변환 (16바이트(128비트) 또는 32바이트(256비트) 키)
    final key = encrypt.Key(keyBytes);
    final iv = encrypt.IV.fromLength(16); // 16바이트 IV 생성
    
    final encrypter = encrypt.Encrypter(encrypt.AES(key, mode: encrypt.AESMode.cbc, padding: 'PKCS7'));
    final encrypted = encrypter.encrypt(plainText, iv: iv);
    
    // IV도 함께 전송해야 함 (이 부분은 서버와 협의 필요)
    final result = {
      'data': encrypted.base64,
      'iv': iv.base64,
    };
    
    return jsonEncode(result);
  }
  
  // AES 복호화 메서드 (향후 사용)
  Future<String> decryptWithAes(String encryptedJson) async {
    final aesKeyBase64 = await getAesKey();
    if (aesKeyBase64 == null) {
      throw Exception('AES 키가 없습니다');
    }
    
    final data = jsonDecode(encryptedJson);
    final encryptedData = data['data'];
    final ivBase64 = data['iv'];
    
    // base64 디코딩해서 바이트 배열로 변환
    final keyBytes = base64Decode(aesKeyBase64);
    
    final key = encrypt.Key(keyBytes);
    final iv = encrypt.IV.fromBase64(ivBase64);
    
    final encrypter = encrypt.Encrypter(encrypt.AES(key, mode: encrypt.AESMode.cbc, padding: 'PKCS7'));
    final decrypted = encrypter.decrypt(encrypt.Encrypted.fromBase64(encryptedData), iv: iv);
    
    return decrypted;
  }
}

// 암호화 서비스 프로바이더
final encryptionServiceProvider = Provider<EncryptionService>((ref) {
  final secureStorage = ref.read(secureStorageProvider);
  return EncryptionService(secureStorage);
});

// AES 키 응답 상태를 관리할 StateNotifier
class AesKeyNotifier extends StateNotifier<AsyncValue<String>> {
  final EncryptApi _encryptApi;
  final EncryptionService _encryptionService;
  
  AesKeyNotifier(this._encryptApi, this._encryptionService) : super(const AsyncValue.data(""));
  
  Future<void> fetchAesKey() async {
    state = const AsyncValue.loading();
    try {
      final response = await _encryptApi.createAesKey();
      String encryptedAesKey = response.data.toString();
      
      // RSA로 암호화된 AES 키를 복호화
      final decryptedAesKey = await _encryptionService.decryptAesKey(encryptedAesKey);
      
      // 복호화된 AES 키를 안전하게 저장
      await _encryptionService.saveAesKey(decryptedAesKey);
      
      state = AsyncValue.data(decryptedAesKey);
    } catch (e) {
      print('AES 키 가져오기 실패: $e');
      state = AsyncValue.error(e, StackTrace.current);
    }
  }
  
  Future<String?> getStoredAesKey() async {
    return await _encryptionService.getAesKey();
  }
}

// AES 키 StateNotifier 프로바이더
final aesKeyNotifierProvider = StateNotifierProvider<AesKeyNotifier, AsyncValue<String>>((ref) {
  final encryptApi = ref.read(encryptApiProvider);
  final encryptionService = ref.read(encryptionServiceProvider);
  return AesKeyNotifier(encryptApi, encryptionService);
});