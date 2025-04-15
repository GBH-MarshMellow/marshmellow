// 앱 전반적인 상수 정의
class AppConstants {
  // 앱 정보(임시)
  static const String APP_NAME = '핀테크 앱';
  static const String APP_VERSION = '1.0.0';
  
  // 시간 관련 상수(임시)
  static const int DEFAULT_TIMEOUT = 30000; // 30초 (밀리초)
  static const int SESSION_TIMEOUT = 1800000; // 30분 (밀리초)
  
  // 페이지네이션 관련(임시)
  static const int DEFAULT_PAGE_SIZE = 20;
  
  // 파일 경로(임시)
  static const String TEMP_DIRECTORY = 'temp';
  static const String DOWNLOAD_DIRECTORY = 'downloads';
  
  // 알림 채널(임시)
  static const String DEFAULT_NOTIFICATION_CHANNEL = 'default_channel';
  
  // 보안 관련(임시)
  static const int PIN_LENGTH = 6;
  static const int MAX_PIN_ATTEMPTS = 5;
  static const int BIOMETRIC_TIMEOUT = 30000; // 30초 (밀리초)
}

// 사용 예시

/*

import 'package:test0316_1/core/constants/app_constants.dart';

// 타임아웃 설정시
dio.options.connectTimeout = Duration(milliseconds: AppConstants.DEFAULT_TIMEOUT);

// 앱 이름 사용시
Text(AppConstants.APP_NAME)

*/