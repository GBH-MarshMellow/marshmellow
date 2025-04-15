import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/data/datasources/remote/certificate_api.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'package:marshmellow/di/providers/core_providers.dart';
import 'package:marshmellow/presentation/viewmodels/auth/certificate_notifier.dart';

class CertificateRepository {
  final CertificateApi _certificateApi;
  final FlutterSecureStorage _secureStorage;

  CertificateRepository(this._certificateApi, this._secureStorage);

  // 통합인증 여부 조회
  Future<bool> hasCompletedIntegratedAuth() async {
    try {
      final response = await _certificateApi.checkIntegratedStatus();
      return response['data'] ?? false;
    } catch (e) {
      print('통합인증 상태 조회 실패: $e');
      return false;
    }
  }

  // 인증서 존재 여부 및 상태 조회
  Future<Map<String, dynamic>> getCertificateStatus() async {
    try {
      final response = await _certificateApi.checkCertificateExist();

      // 서버에서 받은 정보를 secure storage에 저장
      final Map<String, dynamic> certData = response['data'] ??
        {'exist': false, 'status': null, 'certificatePem' : null};

      if (certData['exist'] == true) {
        await _secureStorage.write(
          key: StorageKeys.certificateStatus, 
          value: certData['status']
        );

        // 서버에서 받은 certificatePem 저장
        if (certData['certificatePem'] != null) {
          await _secureStorage.write(
            key: StorageKeys.certificatePem, 
            value: certData['certificatePem']
          );
        }
      }

      return certData;
    } catch (e) {
      print('인증서 상태 조회 실패: $e');

      // API 실패시 저장된 데이터 반환
      final storedStatus = await _secureStorage.read(key: StorageKeys.certificateStatus);
      final storedPem = await _secureStorage.read(key: StorageKeys.certificatePem);

      return {
        'exist': storedPem != null, 
        'status': storedStatus, 
        'certificatePem': storedPem
      };
    }
  }

  // 인증서 발급
  Future<String?> issueCertificate({required String csrPem, required String userEmail}) async {
    try {
      final response = await _certificateApi.issueCertificate(
        csrPem: csrPem, 
        userEmail: userEmail
      );

      final certificatePem = response['data']?['certificatePem'];
      final halfUserKey = response['data']?['halfUserKey'];

      // 발급 성공시 인증서 정보 저장
      if (certificatePem != null) {
        await _secureStorage.write(
          key: StorageKeys.certificatePem, 
          value: certificatePem
        );
        
        await _secureStorage.write(
          key: StorageKeys.certificateStatus, 
          value: 'VALID'
        );
        
        await _secureStorage.write(
          key: StorageKeys.certificateEmail, 
          value: userEmail
        );

        await _secureStorage.write(
          key: StorageKeys.halfUserKey, 
          value: halfUserKey
        );
      }
      return certificatePem;
    } catch (e) {
      print('인증서 발급 실패: $e');
      return null;
    }
  }

  // 저장된 인증서 정보 조회
  Future<Map<String, String?>> getSavedCertificateInfo() async {
    final certificatePem = await _secureStorage.read(key: StorageKeys.certificatePem);
    final certificateStatus = await _secureStorage.read(key: StorageKeys.certificateStatus);
    final certificateEmail = await _secureStorage.read(key: StorageKeys.certificateEmail);

    return {
      'certificatePem': certificatePem,
      'certificateStatus': certificateStatus,
      'certificateEmail': certificateEmail
    };
  }

  // 인증서 비밀번호 저장
  Future<void> saveCertificatePassword(String password) async {
    await _secureStorage.write(
      key: StorageKeys.certificatePassword, 
      value: password
    );
  }

  // 인증서 비밀번호 조회
  Future<String?> getCertificatePassword() async {
    return await _secureStorage.read(key: StorageKeys.certificatePassword);
  }
}

// 프로바이더 정의
final CertificateRepositoryProvider = Provider<CertificateRepository>((ref) {
  final certificateApi = ref.watch(certificateApiProvider);
  final secureStorage = ref.watch(secureStorageProvider);
  return CertificateRepository(certificateApi, secureStorage);
});