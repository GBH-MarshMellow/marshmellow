import 'package:dio/dio.dart';
import 'package:marshmellow/data/datasources/remote/auth_api.dart';

/*
  본인인증 및 sse 연결 repository
*/
class AuthIdentityRepository {
  final AuthApi _authApi;

  AuthIdentityRepository(this._authApi);

  // 본인확인 요청
  Future<Map<String, dynamic>> verifyIdentity(String phoneNumber) async {
    try {
      final response = await _authApi.verifyIdentity(phoneNumber);
      return response;
    } catch (e) {
      if (e is DioException) {
        rethrow;
      }
      throw Exception('본인인증 요청 실패: $e');
    }
  }

  // SSE 연결 URL 가져오기
  String getIdentityVerificationSubscribeURl(String phoneNumber) {
    return _authApi.getIdentityVerificationSubscribeURl(phoneNumber);
  }
}