/*
  mm 인증서 생성 상태
*/
import 'dart:math';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/core/services/advanced_certificate_service.dart';
import 'package:marshmellow/core/services/certificate_service.dart';
import 'package:marshmellow/data/repositories/auth/certificate_repository.dart';
import 'package:marshmellow/di/providers/core_providers.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/mydata/auth_mydata_cert_email_page.dart';

class CertificateProcessState {
  final bool isLoading;
  final String email;
  final String? error;
  final bool isCompleted;
  final String? certificatePem;

  CertificateProcessState({
    this.isLoading = false,
    this.email = '',
    this.error,
    this.isCompleted = false,
    this.certificatePem,
  });

  CertificateProcessState copyWith({
    bool? isLoading,
    String? email,
    String? error,
    bool? isCompleted,
    String? certificatePem,
  }) {
    return CertificateProcessState(
      isLoading: isLoading ?? this.isLoading,
      email: email ?? this.email,
      error: error,
      isCompleted: isCompleted ?? this.isCompleted,
      certificatePem: certificatePem ?? this.certificatePem,
    );
  }
}

/*
  mm 인증서 생성 상태 감지
*/
class CertificateProcessNotifier extends StateNotifier<CertificateProcessState> {
  final CertificateRepository _repository;
  final CertificateService _certificateService;
  final FlutterSecureStorage _secureStorage; // 인증서 비밀번호를 암호화하여 저장합니다

  // 비밀번호는 상태에 저장하지 않고 메모리에만 잠시 저장
  // securestorage에 저장하고 이 메모리는 삭제됩니다
  String _password = '';

  CertificateProcessNotifier(this._repository, this._certificateService, this._secureStorage)
    : super(CertificateProcessState());

  // 이메일 설정
  void setEmail(String email) {
    state = state.copyWith(email: email);
  }

  // 비밀번호 설정 - 메모리에 임시 저장
  void setPassword(String password) {
    _password =  password;
  }

  // mm 인증서 비밀번호 저장 - 암호화하여 저장
  Future<void> saveCertificatePassword() async {
    if (_password.isEmpty) return;

    // FlutterSecureStorage는 저장전 데이터를 암호화합니다
    await _secureStorage.write(
      key: StorageKeys.certificatePassword, 
      value: _password
    );

    // 메모리에서 비밀번호 제거
    _password = '';
  }

    /// 🔐 키쌍이 없을 경우에만 생성
  Future<void> ensureKeyPairExists() async {
    final hasKey = await _certificateService.hasKeyPair();
    if (!hasKey) {
      print('🔐 키쌍 없음 → 생성 시작');
      final keyPair = await _certificateService.generateRSAKeyPair();
      await _certificateService.storeKeyPair(keyPair);
      print('✅ 키쌍 생성 완료');
    } else {
      print('🔐 기존 키쌍 존재 → 재사용');
    }
  }

   // 인증서 발급 요청
  Future<bool> issueCertificate() async {
    print("======== 인증서 발급 프로세스 시작 ========");
    if (state.email.isEmpty) {
      print("오류: 이메일이 비어있음");
      state = state.copyWith(error: '이메일을 입력해주세요.', isLoading: false);
      return false;
    }

    if (_password.isEmpty) {
      print("오류: 비밀번호가 비어있음");
      state = state.copyWith(error: '비밀번호를 설정해주세요.', isLoading: false);
      return false;
    }
    print("상태: 로딩 시작");
    state = state.copyWith(isLoading: true, error: null);

    try {
      // 1. 키쌍 확인
      await ensureKeyPairExists();

      // 2. CSR 생성
      print("📄 CSR 생성 시작");
      final csrPem = await _certificateService.generateCSR(
        commonName: state.email,
        organization: 'GBH',
        country: 'KR',
      );
      print("📄 CSR 생성 결과 (앞): ${csrPem.substring(0, 30)}...");

      // 3. 서버로 인증서 발급 요청
      print("🚀 인증서 발급 요청: ${state.email}");
      final certificatePem = await _repository.issueCertificate(
        csrPem: csrPem,
        userEmail: state.email,
      );
      print("✅ 인증서 발급 응답 수신: ${certificatePem != null}");

      if (certificatePem != null) {
        await saveCertificatePassword();

        await _secureStorage.write(
          key: StorageKeys.certificatePem,
          value: certificatePem,
        );

        state = state.copyWith(
          isLoading: false,
          isCompleted: true,
          certificatePem: certificatePem,
        );
        return true;
      } else {
        state = state.copyWith(
          isLoading: false,
          error: '인증서 발급에 실패했습니다.',
        );
        return false;
      }
    } catch (e, stackTrace) {
      print("❌ 인증서 발급 중 예외: $e");
      print("🧵 스택트레이스: $stackTrace");
      state = state.copyWith(
        isLoading: false,
        error: '인증서 발급 중 오류가 발생했습니다: $e',
      );
      return false;
    }
  }

  // 저장된 인증서 비밀번호 가져오기
  Future<String?> getSaveCertificatePassword() async {
    return await _secureStorage.read(key: StorageKeys.certificatePassword);
  }
    /// 전자서명 또는 기능 사용 가능 여부 체크
  Future<bool> isReadyForSigning() async {
    final hasKey = await _certificateService.hasKeyPair();
    final cert = await _secureStorage.read(key: StorageKeys.certificatePem);
    return hasKey && cert != null;
  }
}

/*
  mm 인증서 생성 프로세스 프로바이더
*/
final certificateProcessProvider = StateNotifierProvider<CertificateProcessNotifier, CertificateProcessState>((ref) {
  final repository = ref.watch(CertificateRepositoryProvider);
  final secureStorage = ref.watch(secureStorageProvider);
  final certificateService = CertificateService(secureStorage);

  // 이메일 값 가져와서 초기 상태 설정
  final email = ref.watch(emailProvider);
  
  final notifier = CertificateProcessNotifier(repository, certificateService, secureStorage);
  notifier.setEmail(email);

  return notifier;
});