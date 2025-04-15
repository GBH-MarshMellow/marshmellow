import 'package:flutter/foundation.dart';
import 'app_config.dart';

class EnvironmentLoader {
  // 환경 설정을 로드하는 정적 메서드
  static Future<void> load() async {
    try {
      // 현재 앱 모드에 따라 환경 결정
      // kDebugMode는 Flutter에서 제공하는 상수로 디버그 모드 여부를 판단
      // 디버그 모드면 개발환경, 아니면 프로덕션 환경으로 설정
      const environment = kDebugMode ? Environment.dev : Environment.prod;
      
      // AppConfig의 환경 설정 초기화 실행
      await AppConfig.initialize(environment);
    } catch (e) {
      // 환경 로딩 중 에러가 발생한 경우 로그 출력
      debugPrint('⚠️ Error loading environment: $e');
      // 에러를 상위로 다시 던져서 처리할 수 있게 함
      rethrow;
    }
  }
}