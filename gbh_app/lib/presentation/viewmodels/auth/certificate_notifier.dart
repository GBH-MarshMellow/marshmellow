/*
  mm 인증서 관련 상태
*/
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/repositories/auth/certificate_repository.dart';

class CertificateState {
  final bool isLoading;
  final bool hasIntegratedAuth;
  final bool hasCertificate;
  final String? certificateStatus;
  final String? certificatePem;
  final String? certificateEmail;
  final String? error;

  CertificateState({
    this.isLoading = false,
    this.hasIntegratedAuth = false,
    this.hasCertificate = false,
    this.certificateStatus,
    this.certificatePem,
    this.certificateEmail,
    this.error,
  });

  CertificateState copyWith({
    bool? isLoading,
    bool? hasIntegratedAuth,
    bool? hasCertificate,
    String? certificateStatus,
    String? certificatePem,
    String? certificateEmail,
    String? error,
  }) {
    return CertificateState(
      isLoading: isLoading ?? this.isLoading,
      hasIntegratedAuth: hasIntegratedAuth ?? this.hasIntegratedAuth,
      hasCertificate: hasCertificate ?? this.hasCertificate,
      certificateStatus: certificateStatus ?? this.certificateStatus,
      certificatePem: certificatePem ?? this.certificatePem,
      certificateEmail: certificateEmail ?? this.certificateEmail,
      error: error,
    );
  }
}

class CertificateNotifier extends StateNotifier<CertificateState> {
  final CertificateRepository _repository;

  CertificateNotifier(this._repository) : super(CertificateState()) {
    // 인증서 정보 로드
    _loadCertificateInfo();
  }

  // 저장된 인증서 정보 로드
  Future<void> _loadCertificateInfo() async {
    final certInfo = await _repository.getSavedCertificateInfo();

    if (certInfo['certificatePem'] != null) {
      state = state.copyWith(
        hasCertificate: true,
        certificateStatus: certInfo['certificateStatus'],
        certificatePem: certInfo['certificatePem'],
        certificateEmail: certInfo['certificateEmail']
      );
    }
  }

  // 통합인증 여부 확인
  Future<void> checkIntegratedAuthStatus() async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final hasIntegratedAuth = await _repository.hasCompletedIntegratedAuth();
      state = state.copyWith(
        isLoading: false,
        hasIntegratedAuth: hasIntegratedAuth,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: '통합인증 상태 확인 중 오류가 발생했습니다: $e'
      );
    }
  }

  // 인증서 존재 유무 확인
  Future<void> checkCertificateStatus() async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final certStatus = await _repository.getCertificateStatus();
      state = state.copyWith(
        isLoading: false,
        hasCertificate: certStatus['exist'] ?? false,
        certificateStatus: certStatus['status'],
        certificatePem: certStatus['certificatePem'],
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: '인증서 상태 확인 중 오류가 발생했습니다: $e',
      );
    }
  }

  // 인증서 발급
  Future<bool> issueCertificate({required String csrPem, required String userEmail}) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final certificatePem = await _repository.issueCertificate(
        csrPem: csrPem,
        userEmail: userEmail,
      );
      
      if (certificatePem != null) {
        state = state.copyWith(
          isLoading: false,
          hasCertificate: true,
          certificatePem: certificatePem,
          certificateStatus: 'VALID',
          certificateEmail: userEmail,
        );
        return true;
      } else {
        state = state.copyWith(
          isLoading: false,
          error: '인증서 발급에 실패했습니다.',
        );
        return false;
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: '인증서 발급 중 오류가 발생했습니다: $e',
      );
      return false;
    }
  }

  // 인증서 비밀번호 저장
  Future<void> saveCertificatePassword(String password) async {
    await _repository.saveCertificatePassword(password);
  }

  // 인증서 비밀번호 가져오기
  Future<String?> getCertificatePassword() async {
    return await _repository.getCertificatePassword();
  }
}

// mm 인증서 프로바이더
final certificateProvider = StateNotifierProvider<CertificateNotifier, CertificateState>((ref) {
  final repository = ref.watch(CertificateRepositoryProvider);
  return CertificateNotifier(repository);
});