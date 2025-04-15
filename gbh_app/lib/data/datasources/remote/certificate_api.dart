import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/data/datasources/remote/api_client.dart';

/*
  mm 인증서 api
*/
class CertificateApi {
  final ApiClient _apiClient;
  final FlutterSecureStorage _secureStorage;

  CertificateApi(this._apiClient, this._secureStorage);
  
  // 통합인증 여부 조회
  Future<Map<String, dynamic>> checkIntegratedStatus() async {
    final response = await _apiClient.get('/api/mm/auth/integrated-status');
    return response.data;
  }

  // 인증서 존재 여부 조회
  Future<Map<String, dynamic>> checkCertificateExist() async {
    final accessToken = await _secureStorage.read(key: StorageKeys.accessToken);

    final response = await _apiClient.get(
      '/api/mm/auth/cert/exist',
      options: Options(
        headers: {
          'Authorization': 'Bearer $accessToken'
        }
      )
    );
    return response.data;
  }

  // 인증서 발급
  Future<Map<String, dynamic>> issueCertificate({
    required String csrPem,
    required String userEmail,
  }) async {
    final response = await _apiClient.post(
      '/api/mm/auth/cert/issue',
      data:{
        'csrPem' : csrPem,
        'userEmail' : userEmail,
      },
    );
    return response.data;
  }
}