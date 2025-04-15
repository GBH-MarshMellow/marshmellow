// remote/encryption/encrypt_api.dart 수정
import 'package:dio/dio.dart';
import 'package:marshmellow/data/datasources/remote/api_client.dart';

class EncryptApi {
  final ApiClient _apiClient;

  EncryptApi(this._apiClient);

  // 암호화된 AES키(암호화 복호화 둘 다 가능한 키)를 받아오기
  Future<Response> createAesKey() async {
    return await _apiClient.post(
      '/api/mm/auth/key-gen',
      options: Options(responseType: ResponseType.plain), // 응답을 plain text로 처리
    );
  }
}