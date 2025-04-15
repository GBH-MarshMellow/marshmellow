import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../../core/config/app_config.dart';
import '../../core/utils/encryption_util.dart';
import 'dart:convert';

// SharedPreferences 프로바이더
final sharedPreferencesProvider = Provider<SharedPreferences?>((ref) {
  // null로 초기화하고 나중에 main.dart에서 오버라이드
  return null;
});

// SecureStorage 프로바이더
final secureStorageProvider = Provider<FlutterSecureStorage>((ref) {
  return FlutterSecureStorage(
      aOptions: AndroidOptions(encryptedSharedPreferences: true));
});

// 암호화 유틸리티 프로바이더
final encryptionUtilProvider = Provider<EncryptionUtil>((ref) {
  final secureStorage = ref.read(secureStorageProvider);
  return EncryptionUtil(secureStorage);
});

// <<<<<<<<<<<< [ T E S T - Token 4월2일 만료 ] <<<<<<<<<<<<<<<<<<<<<<<<

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


// Dio 프로바이더
final dioProvider = Provider<Dio>((ref) {
  final dio = Dio(BaseOptions(
    baseUrl: AppConfig.apiBaseUrl,
    connectTimeout: const Duration(seconds: 30),
    receiveTimeout: const Duration(seconds: 30),
    sendTimeout: const Duration(seconds: 30),
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      // <<<<<<<<<<<< [ T E S T - Token 4월2일 만료 ] <<<<<<<<<<<<<<<<<<<<<<<<

      // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    },
    validateStatus: (status) {
      return status! < 500; // 서버 에러만 예외 처리
    },
  ));

  // 암호화 유틸리티
  final encryptionUtil = ref.read(encryptionUtilProvider);

  dio.interceptors.add(InterceptorsWrapper(
    // 요청 인터셉터 추가 - accessToken 처리
    onRequest: (options, handler) async {
      // options.extra에서 requiresAuth 값 확인(default: true입니다)
      final requiresAuth = options.extra['requiresAuth'] ?? true;

      if (requiresAuth && !options.headers.containsKey('Authorization')) {
        final secureStorage = ref.read(secureStorageProvider);
        final token = await secureStorage.read(key: 'access_token');
        if (token != null) {
          options.headers['Authorization'] = 'Bearer $token';
        }
      }
      // 자산 API 요청인 경우 암호화 처리
      if (encryptionUtil.isAssetApiPath(options.path)) {
        try {
          // 요청 데이터가 있는 경우만 처리
          if (options.data != null) {
            print('암호화 전 요청 데이터: ${options.data}');
            Map<String, dynamic> requestData;

            // 문자열인 경우 JSON으로 파싱
            if (options.data is String) {
              requestData = jsonDecode(options.data);
            }
            // Map인 경우 그대로 사용
            else if (options.data is Map) {
              requestData = Map<String, dynamic>.from(options.data);
            }
            // 다른 타입인 경우 변환
            else {
              requestData = Map<String, dynamic>.from(options.data);
            }

            // 요청 데이터 암호화
            final encryptedData =
                await encryptionUtil.encryptRequest(requestData);
            options.data = encryptedData;
            print('암호화 후 요청 데이터: ${options.data}');
          }
        } catch (e) {
          print('요청 암호화 오류: $e');
          return handler.reject(DioException(
            requestOptions: options,
            error: '요청 암호화 실패: $e',
          ));
        }
      }

      return handler.next(options);
    },

    // 응답 인터셉터 추가 - 백엔드 커스텀 상태 코드 처리
    onResponse: (response, handler) async {
      // 자산 API 응답인 경우 복호화 처리
      if (encryptionUtil.isAssetApiPath(response.requestOptions.path)) {
        try {
          // 응답 데이터가 Map인 경우 복호화 시도
          if (response.data is Map) {
            final Map<String, dynamic> responseData =
                Map<String, dynamic>.from(response.data);
            final decryptedData =
                await encryptionUtil.decryptResponse(responseData);
            response.data = decryptedData;
          }
        } catch (e) {
          print('응답 복호화 오류: $e');
          return handler.reject(DioException(
            requestOptions: response.requestOptions,
            response: response,
            error: '응답 복호화 실패: $e',
          ));
        }
      }
      // 응답에서 실제 상태 코드 확인 (response.data.status)
      if (response.data is Map && response.data['status'] != null) {
        final statusCode = response.data['status'];
        // 성공 상태 코드 범위 확인 (예: 200~299)
        if (statusCode < 200 || statusCode >= 300) {
          // 에러 응답으로 변환
          return handler.reject(
            DioException(
              requestOptions: response.requestOptions,
              response: response,
              error: '백엔드 에러: 상태 코드 ${response.data['status']}',
            ),
          );
        }
      }
      return handler.next(response);
    },
  ));

  // 디버그 모드일 때만 로그 인터셉터 추가
  if (AppConfig.debugMode) {
    dio.interceptors.add(LogInterceptor(
      requestBody: true,
      responseBody: true,
    ));
  }

  return dio;
});
