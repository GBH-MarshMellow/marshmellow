// import 'dart:async';
// import 'package:flutter/services.dart';
// import 'package:flutter_secure_storage/flutter_secure_storage.dart';
// import 'certificate_service.dart'; // 기존의 Flutter 구현

// /*
//  TEE/SE 하드웨어 접근 가능 여부를 확인하고
//  가능한 경우 코틀린 / 불가능한 경우 플러터로 하이브리드 접근
// */
// class AdvancedCertificateService {
//   static const MethodChannel _channel = MethodChannel('com.gbh.marshmellow/secure_keys');
//   final CertificateService _flutterService;
//   bool _isHardwareSecurityAvailable = false;
  
//   AdvancedCertificateService(FlutterSecureStorage secureStorage) 
//       : _flutterService = CertificateService(secureStorage) {
//     // 초기화 시 하드웨어 보안 지원 여부 확인
//     _checkHardwareSupport();
//   }
  
//   Future<void> _checkHardwareSupport() async {
//     try {
//       final result = await _channel.invokeMethod<Map<dynamic, dynamic>>('isHardwareSecurityAvailable');
//       _isHardwareSecurityAvailable = result?['available'] ?? false;
//       print('하드웨어 보안 지원: $_isHardwareSecurityAvailable');
//     } on PlatformException catch (e) {
//       print('하드웨어 보안 확인 실패: ${e.message}');
//       _isHardwareSecurityAvailable = false;
//     }
//   }
  
//   // 키 쌍 생성
//   Future<bool> generateKeyPair() async {
//     if (_isHardwareSecurityAvailable) {
//       // 하드웨어 TEE/SE 사용
//       try {
//         final result = await _channel.invokeMethod<Map<dynamic, dynamic>>('generateKeyPair');
//         return result?['success'] ?? false;
//       } on PlatformException catch (e) {
//         print('하드웨어 기반 키 생성 실패, 소프트웨어 방식으로 전환: ${e.message}');
//         // 하드웨어 방식 실패 시 Flutter 구현으로 폴백
//         final keyPair = await _flutterService.generateRSAKeyPair();
//         await _flutterService.storeKeyPair(keyPair);
//         return true;
//       }
//     } else {
//       // 하드웨어 지원이 없으면 Flutter 구현 사용
//       final keyPair = await _flutterService.generateRSAKeyPair();
//       await _flutterService.storeKeyPair(keyPair);
//       return true;
//     }
//   }
  
//   // CSR 생성
//   Future<String?> generateCSR({
//     required String commonName,
//     String organization = 'GBH',
//     String country = 'KR'
//   }) async {
//     return _flutterService.generateCSR(
//           commonName: commonName,
//           organization: organization,
//           country: country
//         );
//     // if (_isHardwareSecurityAvailable) {
//     //   try {
//     //     final result = await _channel.invokeMethod<Map<dynamic, dynamic>>(
//     //       'generateCSR',
//     //       {
//     //         'commonName': commonName,
//     //         'organization': organization,
//     //         'country': country
//     //       }
//     //     );
//     //     return result?['csr'];
//     //   } on PlatformException catch (e) {
//     //     print('하드웨어 CSR 생성 실패, 소프트웨어 방식으로 전환: ${e.message}');
//     //     // 하드웨어 방식 실패 시 Flutter 구현으로 폴백
//     //     return _flutterService.generateCSR(
//     //       commonName: commonName,
//     //       organization: organization,
//     //       country: country
//     //     );
//     //   }
//     // } else {
//     //   // 하드웨어 지원이 없으면 Flutter 구현 사용
//     //   return _flutterService.generateCSR(
//     //     commonName: commonName,
//     //     organization: organization,
//     //     country: country
//     //   );
//     // }
//   }
  
//   // 키 쌍 존재 여부 확인 
//   Future<bool> hasKeyPair() async {
//     if (_isHardwareSecurityAvailable) {
//       try {
//         final result = await _channel.invokeMethod<Map<dynamic, dynamic>>('hasKeyPair');
//         return result?['exists'] ?? false;
//       } on PlatformException catch (e) {
//         print('하드웨어 키 확인 실패, 소프트웨어 방식으로 전환: ${e.message}');
//         return _flutterService.hasKeyPair();
//       }
//     } else {
//       return _flutterService.hasKeyPair();
//     }
//   }
  
//   // 서명 기능
//   Future<String?> signData(String data) async {
//     if (_isHardwareSecurityAvailable) {
//       try {
//         final result = await _channel.invokeMethod<Map<dynamic, dynamic>>(
//           'signData',
//           {'data': data}
//         );
//         return result?['signature'];
//       } on PlatformException catch (e) {
//         print('하드웨어 서명 실패, 소프트웨어 방식으로 전환: ${e.message}');
//         // 소프트웨어 방식 서명은 별도 구현 필요
//         // 현재 Flutter 코드에 서명 메서드가 분리되어 있지 않음
//         return null;
//       }
//     } else {
//       // 소프트웨어 방식 서명 구현 필요
//       return null;
//     }
//   }
  
//   // 하드웨어 보안 사용 여부 확인
//   bool isUsingHardwareSecurity() {
//     return _isHardwareSecurityAvailable;
//   }
// }