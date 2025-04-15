// data/datasources/remote/auth_api.dart
import 'package:dio/dio.dart';
import 'package:marshmellow/data/datasources/remote/api_client.dart';

class AuthApi {
  final ApiClient _apiClient;
  
  AuthApi(this._apiClient);

  // 본인확인 api
  Future<Map<String, dynamic>> verifyIdentity(String phoneNumber) async {
    final response = await _apiClient.post(
      '/api/mm/auth/identity-verify',
      requiresAuth: false,
      data: {
        'phoneNumber' : phoneNumber,
      }
    );
    return response.data;
  }

  // SSE 연결을 위한 URL 반환
  String getIdentityVerificationSubscribeURl(String phoneNumber) {
    return '/api/mm/auth/subscribe/$phoneNumber';
  }

  // 핀번호 회원가입 api
  Future<Map<String, dynamic>> signUp({
    required String userName,
    required String phoneNumber,
    required String userCode,
    required String pin,
    required String fcmToken
  }) async {
    final response = await _apiClient.post(
      '/api/mm/auth/sign-up',
      requiresAuth: false,
      data: {
      'userName': userName,
      'phoneNumber': phoneNumber,
      'userCode': userCode,
      'pin': pin,
      'fcmToken' : fcmToken,  
      },
      options: Options(
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        }
      )
    );
    return response.data;
  }

  // 핀번호 로그인 api
  Future<Map<String, dynamic>> loginWithPin({
    required String phoneNumber,
    required String pin,
  }) async {
    final response = await _apiClient.post(
      '/api/mm/auth/login/pin',
      requiresAuth: false,
      data: {
      'phoneNumber': phoneNumber,
      'pin': pin
      }
    );
    return response.data;
  }

  // 핀번호 생체 로그인 api
  Future<Map<String, dynamic>> loginWithBiometrics({
    required String phoneNumber
  }) async {
    final response = await _apiClient.post(
      '/api/mm/auth/login/bio',
      requiresAuth: false,
      data: {
        'phoneNumber': phoneNumber
      }
    );
    return response.data;
  }

  // accessToken 재발급
  Future<Map<String, dynamic>> reissueToken({
    required String refreshToken
  }) async {
    final response = await _apiClient.post(
      '/api/mm/auth/reissue',
      requiresAuth: false,
      data: {
        'refreshToken': refreshToken
      }
    );
    return response.data;
  }

  // 핀번호 로그아웃
  Future<Map<String, dynamic>> logout() async {
    final response = await _apiClient.post('/api/mm/auth/logout');
    return response.data;
  }
}