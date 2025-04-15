// lib/data/datasources/remote/api_client.dart
import 'package:dio/dio.dart';

class ApiClient {
  final Dio _dio;

  ApiClient(this._dio);

  // 기본 GET 요청
  Future<dynamic> get(String path,{
    Map<String, dynamic>? queryParameters,
    Options? options,
    bool requiresAuth = true,
  }) async {
    try {
      final finalOptions = options ?? Options();
      finalOptions.extra = finalOptions.extra ?? {};
      finalOptions.extra!['requiresAuth'] = requiresAuth;

      final response = await _dio.get(
        path, 
        queryParameters: queryParameters,
        options: finalOptions
      );
      return _processResponse(response);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // GET 요청에 body를 지원하는 메소드
  Future<dynamic> getWithBody(String path,
      {dynamic data, 
      Map<String, dynamic>? queryParameters,
      Options? options,
      bool requiresAuth = true}) async {
    try {
      final finalOptions = options ?? Options(method: 'GET');
      finalOptions.extra = finalOptions.extra ?? {};
      finalOptions.extra!['requiresAuth'] = requiresAuth;

      final response = await _dio.request(
        path,
        data: data,
        queryParameters: queryParameters,
        options: finalOptions,
      );
      return _processResponse(response);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // 기본 POST 요청
  Future<dynamic> post(
    String path, {
    dynamic data,
    Options? options,
    bool requiresAuth = true,
  }) async {
    try {
      final finalOptions = options ?? Options();
      finalOptions.extra ??= {};
      finalOptions.extra!['requiresAuth'] = requiresAuth;

      final response = await _dio.post(
        path, 
        data: data,
        options: finalOptions,
      );
      return _processResponse(response);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // PUT 요청 추가
  Future<dynamic> put(
    String path, {
    dynamic data,
    Options? options,
    bool requiresAuth = true,
  }) async {
    try {
      final finalOptions = options ?? Options();
      finalOptions.extra ??= {};
      finalOptions.extra!['requiresAuth'] = requiresAuth;

      final response = await _dio.put(
        path, 
        data: data,
        options: finalOptions,
      );
      return _processResponse(response);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // PATCH 요청 추가
  Future<dynamic> patch(
    String path, {
    dynamic data,
    Options? options,
    bool requiresAuth = true,
  }) async {
    try {
      final finalOptions = options ?? Options();
      finalOptions.extra ??= {};
      finalOptions.extra!['requiresAuth'] = requiresAuth;

      final response = await _dio.patch(
        path, 
        data: data,
        options: finalOptions,
      );
      return _processResponse(response);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // DELETE 요청 추가
  Future<dynamic> delete(
    String path, {
    dynamic data,
    Options? options,
    bool requiresAuth = true,
  }) async {
    try {
      final finalOptions = options ?? Options();
      finalOptions.extra ??= {};
      finalOptions.extra!['requiresAuth'] = requiresAuth;

      final response = await _dio.delete(
        path, 
        data: data,
        options: finalOptions,
      );
      return _processResponse(response);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // 응답 처리 (백엔드의 status 코드 확인)
  dynamic _processResponse(Response response) {
    return response;
  }

  // 에러 핸들링 (상세)
  Exception _handleError(DioException error) {
    if (error.response != null) {
      // 응답이 있는 경우
      if (error.response!.data is Map) {
        final Map responseData = error.response!.data;
        if (responseData.containsKey('message')) {
          // 백엔드에서 제공하는 에러 메시지가 있으면 사용
          return Exception('API 에러: ${responseData['message']}');
        }
        if (responseData.containsKey('status')) {
          // 상태 코드가 있으면 포함
          return Exception('API 에러: 상태 코드 ${responseData['status']}');
        }
      }
      return Exception('API 에러: 상태 코드 ${error.response!.statusCode}');
    }

    // 네트워크 연결 문제 등
    if (error.type == DioExceptionType.connectionTimeout ||
        error.type == DioExceptionType.receiveTimeout ||
        error.type == DioExceptionType.sendTimeout) {
      return Exception('서버 연결 시간 초과. 네트워크 상태를 확인해주세요.');
    }

    return Exception('API 요청 실패: ${error.message}');
  }
}