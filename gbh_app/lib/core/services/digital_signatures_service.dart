import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/core/services/certificate_service.dart';
import 'dart:convert';
/*
  전자서명 생성 및 검증 서비스
*/
class DigitalSignatureService {
  final CertificateService _certificateService;
  final FlutterSecureStorage _secureStorage;

  DigitalSignatureService(this._certificateService, this._secureStorage);

  // 반쪽키 가져오기
  Future<String?> getHalfUserKey() async {
    final savedHalfUserKey = await _secureStorage.read(key: StorageKeys.halfUserKey);
    print('전자서명검증에사용할반쪽키는: $savedHalfUserKey');
    
    return savedHalfUserKey;
  }



  // 전자서명 생성
  Future<Map<String, dynamic>> generateDigitalSignature(String originalText) async {
    try {
      // 1. 반쪽 키 가져오기
      final halfUserKey = await getHalfUserKey();
      print('$halfUserKey');

      final normalizedText = originalText
      .replaceAll(RegExp(r'[\u00A0\u200B\u202F]'), ' ') // ← 특수 스페이스 제거
      .replaceAll('\r\n', '\n')
      .replaceAll('\r', '')
      .replaceAll(RegExp(r'\s+'), ' ') // ← 여러 공백 하나로
      .trim();
      print("📦 normalizedText:");
      print(normalizedText);

      final bytes = utf8.encode(normalizedText);
      print("📦 클라 원문 바이트: $bytes");
      print("📦 클라 원문 바이트 길이: ${bytes.length}");
      print("🔑 클라 검증용 원문(Base64) 바이트 길이: ${base64.encode(bytes).length}");
      print("🔑 클라 검증용 원문(Base64): ${base64.encode(bytes)}");
      
      // 2. 원문 서명
      final signedData = await _certificateService.signData(normalizedText);
      
      if (signedData == null) {
        throw Exception('전자서명 생성에 실패했습니다.');
      }
      print('$signedData');
      print('🔑 서명용 원문(Base64): ${base64.encode(utf8.encode(normalizedText))}');
      
      // 3. 인증서 가져오기
      final certificatePemRaw = await _secureStorage.read(key: StorageKeys.certificatePem);
      if (certificatePemRaw == null) {
        throw Exception('인증서를 찾을 수 없습니다.');
      }
      // 인증서에 \r\n 줄바꿈 유지 (기존 줄바꿈 제거 로직 변경)
      final certificatePem = certificatePemRaw
          .replaceAll('\n', '\r\n')  // 일반 줄바꿈을 CRLF로 변환
          .replaceAll('\r\r\n', '\r\n').trim(); 

      print('원본 인증서 데이터: $certificatePemRaw');
      print('처리된 인증서 데이터: $certificatePem');
      if (certificatePem == null) {
        throw Exception('인증서를 찾을 수 없습니다.');
      }
      // print('서버에 이거 보냄 : $certificatePem');

      print('원본 데이터: $certificatePemRaw');
      print('처리된 데이터: $certificatePem');

      
      // 4. 은행 및 카드사 코드 리스트
      final orgList = [
        "BOK", "KDB", "IBK", "KB", "NH", "WOORI", "SC", "CITI", "DGB", "KJB",
        "JJB", "JB", "BNK", "MG", "HANA", "SHINHAN", "KAKAO", "SSAFY",
        "1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"
      ];
      
      // 반환할 데이터 패키지
      return {
        'signedData': signedData,
        'originalText': normalizedText,
        'halfUserKey': halfUserKey,
        'certificatePem': certificatePem,
        'orgList': orgList
      };
    } catch (e) {
      print('전자서명 생성 실패: $e');
      rethrow;
    }
  }
}