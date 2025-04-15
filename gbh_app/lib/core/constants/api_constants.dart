// API 관련 상수 정의
class ApiConstants {
  
  // API 상태 코드(임시)
  static const int SUCCESS = 200;
  static const int CREATED = 201;
  static const int BAD_REQUEST = 400;
  static const int UNAUTHORIZED = 401;
  static const int FORBIDDEN = 403;
  static const int NOT_FOUND = 404;
  static const int SERVER_ERROR = 500;
  
  // 헤더 키(임시)
  static const String CONTENT_TYPE = 'Content-Type';
  static const String AUTHORIZATION = 'Authorization';
  static const String ACCEPT = 'Accept';
  
  // 헤더 값(임시)
  static const String APPLICATION_JSON = 'application/json';
  static const String BEARER = 'Bearer';
}

// 사용 예시

/*

import 'package:test0316_1/core/constants/api_constants.dart';

// Dio 또는 Http 요청시
final response = await dio.post(
  '${AppConfig.apiBaseUrl}${ApiConstants.LOGIN}',
  data: loginData
);

// 응답 상태 체크
if (response.statusCode == ApiConstants.SUCCESS) {
  // 성공 처리
}

*/