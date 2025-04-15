// 로컬 저장소 키 상수 정의
class StorageKeys {
  // SharedPreferences 키
  static const String accessToken = 'access_token';
  static const String refreshToken = 'refresh_token';
  static const String userId = 'user_id';
  static const String isLoggedIn = 'is_logged_in';
  static const String lastLoginTime = 'last_login_time';
  static const String appTheme = 'app_theme';
  static const String languageCode = 'language_code';
  
  // Secure Storage 키
  static const String encryptedPin = 'encrypted_pin';
  static const String privateKey = 'private_key';
  static const String certificate = 'certificate';
  static const String useBiometrics = 'use_biometrics';
  static const String userName = 'user_name';
  static const String phoneNumber = 'phone_number'; 
  static const String userCode = 'user_code'; // 주민번호앞7자리
  static const String carrier = 'user_carrier'; // 통신사정보
  static const String halfUserKey = 'half_user_key';
  static const String aesKey = 'aes_key';
  static const String aesIv = 'aes_iv';
  static const String userkey = 'user_key';

  // mm 인증서 관련 키
  static const String certificatePassword = 'certificate_password';
  static const String certificatePem = 'certificate_pem'; // MM인증서
  static const String certificateStatus = 'certificate_status';
  static const String certificateEmail = 'certificate_email';
  
  // Hive Box 이름
  static const String userBox = 'user_box';
  static const String settingsBox = 'settings_box';
  static const String transactionBox = 'transaction_box';
}

// 사용 예시

/*

import 'package:marshmellow/core/constants/storage_keys.dart';

// SharedPreferences 사용시
final prefs = await SharedPreferences.getInstance();
final token = prefs.getString(StorageKeys.accessToken);

// Secure Storage 사용시
final storage = FlutterSecureStorage();
await storage.write(key: StorageKeys.encryptedPin, value: encryptedPin);

// 생체인식 설정 저장
await storage.write(key: StorageKeys.useBiometrics, value: useBiometrics.toString());

// 사용자 전화번호 저장
await storage.write(key: StorageKeys.phoneNumber, value: phoneNumber);

*/