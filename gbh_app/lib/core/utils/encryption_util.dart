// lib/core/utils/encryption_util.dart

import 'dart:convert';
import 'package:encrypt/encrypt.dart' as encrypt;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';

/// 암호화 관련 유틸리티 클래스
class EncryptionUtil {
  final FlutterSecureStorage _secureStorage;
  
  EncryptionUtil(this._secureStorage);
  
  /// 자산 API 경로인지 확인하는 메서드
  bool isAssetApiPath(String path) {
    // '/asset'으로 시작하는 경로인지 확인
    return path.startsWith('/asset');
  }
  
  /// AES 키를 안전하게 가져오는 메서드
  Future<String?> getAesKey() async {
    return await _secureStorage.read(key: StorageKeys.aesKey);
  }
  
  /// 요청 데이터 암호화
  /// Map 형태의 요청 데이터 중 암호화가 필요한 필드를 암호화하고 IV를 추가
  Future<Map<String, dynamic>> encryptRequest(Map<String, dynamic> requestData) async {
    final aesKeyBase64 = await getAesKey();
    if (aesKeyBase64 == null) {
      throw Exception('AES 키가 없습니다. 먼저 AES 키를 가져와주세요.');
    }
    
    // IV 생성
    final iv = encrypt.IV.fromSecureRandom(16); // 16바이트(128비트) IV
    final ivBase64 = iv.base64;
    
    // AES 키 설정
    final keyBytes = base64Decode(aesKeyBase64);
    final key = encrypt.Key(keyBytes);
    final encrypter = encrypt.Encrypter(
      encrypt.AES(key, mode: encrypt.AESMode.cbc, padding: 'PKCS7')
    );
    
    // 암호화할 필드 식별 및 암호화
    final encryptedData = Map<String, dynamic>.from(requestData);
    
    // IV 추가
    encryptedData['iv'] = ivBase64;
    
    // 항상 암호화해야 하는 민감 필드들 목록
    // final sensitiveFields = ['cardNo', 'accountNo', 'cvc', 'password'];
    final sensitiveFields = ['authCode','cardNo', 'accountNo', 'cvc', 'password','depositAccountNo','transactionSummary','transactionBalance'];
    
    // 민감 필드 암호화
    sensitiveFields.forEach((field) {
      if (encryptedData.containsKey(field) && encryptedData[field] is String) {
        print('암호화 대상 필드: $field = ${encryptedData[field]}');
        final encrypted = encrypter.encrypt(encryptedData[field], iv: iv);
        encryptedData[field] = encrypted.base64;
        print('암호화 후: $field = ${encryptedData[field]}');
      }
    });
    
    // 디버깅
    print('암호화된 요청 데이터: $encryptedData');
    
    return encryptedData;
  }
  
  /// 응답 데이터 복호화
  /// Map 형태의 응답 데이터 중 암호화된 필드를 복호화
  Future<Map<String, dynamic>> decryptResponse(Map<String, dynamic> responseData) async {
    // data 필드가 없으면 그대로 반환
    if (!responseData.containsKey('data') || responseData['data'] == null) {
      return responseData;
    }
    
    final data = responseData['data'];
    
    // data가 Map이 아니면 그대로 반환
    if (data is! Map<String, dynamic>) {
      return responseData;
    }
    
    // IV가 없으면 그대로 반환
    if (!data.containsKey('iv') || data['iv'] == null) {
      return responseData;
    }
    
    final aesKeyBase64 = await getAesKey();
    if (aesKeyBase64 == null) {
      throw Exception('AES 키가 없습니다. 먼저 AES 키를 가져와주세요.');
    }
    
    // AES 키 설정
    final keyBytes = base64Decode(aesKeyBase64);
    final key = encrypt.Key(keyBytes);
    final ivBase64 = data['iv'];
    final iv = encrypt.IV.fromBase64(ivBase64);
    final encrypter = encrypt.Encrypter(
      encrypt.AES(key, mode: encrypt.AESMode.cbc, padding: 'PKCS7')
    );
    
    // 복사본 생성
    final decryptedData = Map<String, dynamic>.from(data);
    
    // 재귀적으로 모든 필드를 확인하며 복호화
    _decryptFields(decryptedData, encrypter, iv);
    
    // 복호화된 데이터로 교체
    final result = Map<String, dynamic>.from(responseData);
    result['data'] = decryptedData;
    
    return result;
  }
  
  /// 재귀적으로 모든 필드를 확인하며 암호화된 필드 복호화
  void _decryptFields(Map<String, dynamic> data, encrypt.Encrypter encrypter, encrypt.IV iv) {
    data.forEach((key, value) {
      if (value is Map<String, dynamic>) {
        // 중첩된 맵인 경우 재귀적으로 처리
        _decryptFields(value, encrypter, iv);
      } else if (value is List) {
        // 리스트인 경우 각 항목 처리
        for (int i = 0; i < value.length; i++) {
          if (value[i] is Map<String, dynamic>) {
            _decryptFields(value[i], encrypter, iv);
          }
        }
      } else if (value is String && _isEncryptedField(key, value)) {
        // 암호화된 필드로 추정되는 경우 복호화 시도
        try {
          final decrypted = encrypter.decrypt(encrypt.Encrypted.fromBase64(value), iv: iv);
          data[key] = decrypted;
        } catch (e) {
          // 복호화 실패 시 원래 값 유지 (암호화된 필드가 아닐 수 있음)
          print('필드 복호화 실패: $key - $e');
        }
      }
    });
  }
  
  /// 암호화된 필드인지 확인하는 메서드
  bool _isEncryptedField(String key, String value) {
    // 암호화된 필드를 식별하는 로직
    // 1. 'iv' 키는 제외
    if (key == 'iv') return false;
    
    // 2. 'encode' 또는 'encrypted'가 들어간 키는 암호화된 필드로 간주
    final alwaysEncryptedFields = ['authCode', 'cardNo', 'accountNo','cvc'];
    if (alwaysEncryptedFields.contains(key)) {
      return true;
    }
    
    // 3. 'Balance', 'amount', 'No' 등의 민감한 정보 키워드가 포함된 경우
    final sensitiveKeywords = ['balance', 'amount', 'no', 'date', 'time', 'type', 'summary', 'memo', 'merchant', 'category', 'bill', 'status','cardNo','installment','authCode'];
    for (var keyword in sensitiveKeywords) {
      if (key.toLowerCase().contains(keyword)) {
        // 값이 Base64 형식인지 확인 (간단한 휴리스틱)
        if (_looksLikeBase64(value)) {
          return true;
        }
      }
    }
    
    return false;
  }
  
  /// 문자열이 Base64 형식인지 간단히 체크
  bool _looksLikeBase64(String value) {
    // 암호화된 값은 대개 Base64 형식을 따르며 특정 길이 이상임
    if (value.length < 16) return false; // 너무 짧으면 암호화된 값이 아님
    
    // Base64 패턴 체크 (선택적)
    final base64Regex = RegExp(r'^[A-Za-z0-9+/]*={0,2}$');
    return base64Regex.hasMatch(value);
  }
}