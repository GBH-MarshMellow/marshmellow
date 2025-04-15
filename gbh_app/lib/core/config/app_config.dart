import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

enum Environment {
  dev,
  prod,
}

class AppConfig {
  // 사용할 변수들 불러오기
  static late final Environment _environment;
  static late final String apiBaseUrl;
  static late final bool debugMode;
  static late final String rsaPrivateKey;
  // static late final String testVariable; // 새로운 변수 추가 예시 (수정가능)(수정가능)(수정가능)(수정가능)

  // 환경 설정을 초기화하는 메서드
  static Future<void> initialize(Environment env) async {
    _environment = env;

    // 어떤 환경 파일을 로드할지 결정
    String fileName;
    switch (env) {
      case Environment.dev:
        fileName = '.env.dev';
        break;
      case Environment.prod:
        fileName = '.env.prod';
        break;
    }
    
    // dotenv를 사용해 해당 환경 파일 로드
    await dotenv.load(fileName: fileName);
    
    // 환경 변수 값들 변수에 저장
    apiBaseUrl = dotenv.get('API_BASE_URL');
    // testVariable = dotenv.get('TEST_VARIABLE'); // 새로운 변수 추가 예시 (수정가능)(수정가능)(수정가능)(수정가능)
    debugMode = dotenv.get('DEBUG_MODE') == 'true';
    // RSA 개인키 저장
    rsaPrivateKey = "-----BEGIN PRIVATE KEY-----\n" +
                    dotenv.get('RSA_PRIVATE_KEY') +
                    "\n-----END PRIVATE KEY-----";


    // 초기화 로그 출력 (어떤 환경으로 시작되었는지)
    debugPrint('🚀 App initialized with ${env.name} environment');
    debugPrint('🔗 API URL: $apiBaseUrl');
  }
  

  // 현재 개발 환경인지 확인하는 헬퍼 메서드
  static bool isDevelopment() => _environment == Environment.dev;
  // 현재 프로덕션 환경인지 확인하는 헬퍼 메서드
  static bool isProduction() => _environment == Environment.prod;
}