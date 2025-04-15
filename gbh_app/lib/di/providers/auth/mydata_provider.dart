import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/repositories/auth/digital_signature_repository.dart';
import 'package:marshmellow/di/providers/auth/certificate_process_provider.dart';
import 'package:marshmellow/presentation/viewmodels/auth/certificate_notifier.dart';

// 임시 비밀번호 저장용 프로바이더
final previousPasswordProvider = StateProvider<String>((ref) => '');

/*
  mm 인증서 비밀번호 상태 
*/
class MydataPasswordState {
  final String password;
  final int currentDigit;
  final bool isConfirmingPassword;
  final bool isLoading;
  final String? error;

  MydataPasswordState({
    this.password = '',
    this.currentDigit = 0,
    this.isConfirmingPassword = false,
    this.isLoading = false,
    this.error,
  });

  MydataPasswordState copyWith({
    String? password,
    int? currentDigit,
    bool? isConfirmingPassword,
    bool? isLoading,
    String? error,
  }) {
    return MydataPasswordState(
      password: password ?? this.password,
      currentDigit: currentDigit ?? this.currentDigit,
      isConfirmingPassword: isConfirmingPassword ?? this.isConfirmingPassword,
      isLoading: isLoading ?? this.isLoading,
      error: error,
    );
  }
}

/*
  mm 인증서 로그인 상태
*/
class MydataLoginState {
  final String password;
  final int currentDigit;
  final bool isLoading;
  final String? error;

  MydataLoginState({
    this.password = '',
    this.currentDigit = 0,
    this.isLoading = false,
    this.error,
  });

  MydataLoginState copyWith({
    String? password,
    int? currentDigit,
    bool? isLoading,
    String? error,
  }) {
    return MydataLoginState(
      password: password ?? this.password,
      currentDigit: currentDigit ?? this.currentDigit,
      isLoading: isLoading ?? this.isLoading,
      error: error,
    );
  }
}

/*
  전자서명 동의 상태
*/
class AgreementState {
  final bool firstAgreement;
  final bool secondAgreement;
  final bool isAtBottom;

  AgreementState({
    this.firstAgreement = false,
    this.secondAgreement = false,
    this.isAtBottom = false,   
  });
  bool get isButtonEnabled => firstAgreement && secondAgreement && isAtBottom;

  AgreementState copyWith({
    bool? firstAgreement,
    bool? secondAgreement,
    bool? isAtBottom,
  }) {
    return AgreementState(
      firstAgreement: firstAgreement ?? this.firstAgreement,
      secondAgreement: secondAgreement ?? this.secondAgreement,
      isAtBottom: isAtBottom ?? this.isAtBottom,
    );
  }
}

/*
  마이데이터 동의 상태
*/
class MydataAgreementState {
  final bool isLoading;
  final bool isCompleted;
  final String? error;
  final String? userKey; // 마이데이터 통합 키 (서버에서 반환)
  final bool isVerified; // 전자서명 검증 여부

  MydataAgreementState({
    this.isLoading = false,
    this.isCompleted = false,
    this.error,
    this.userKey,
    this.isVerified = false,
  });

  MydataAgreementState copyWith({
    bool? isLoading,
    bool? isCompleted,
    String? error,
    String? userKey,
    bool? isVerified,
  }) {
    return MydataAgreementState(
      isLoading: isLoading ?? this.isLoading,
      isCompleted: isCompleted ?? this.isCompleted,
      error: error,
      userKey: userKey ?? this.userKey,
      isVerified: isVerified ?? this.isVerified,
    );
  }
}


/*
  mm 인증서 비밀번호 상태 감지
*/
class MydataPasswordNotifier extends StateNotifier<MydataPasswordState> {
  final Ref _ref;

  MydataPasswordNotifier(this._ref) : super(MydataPasswordState());

  // 비밀번호 초기화
  void resetPassword() {
    state = state.copyWith(password: '', currentDigit: 0);
  }

  // 새로운 숫자 추가
  void addDigit(String digit) {
    if (state.currentDigit < 6) {
      final newPassword = state.password + digit;
      state = state.copyWith(
        password: newPassword,
        currentDigit: state.currentDigit + 1,
      );
    }
  }

  // 확인 모드 설정
  void setConfirmMode(bool isConfirming) {
    state = state.copyWith(
      isConfirmingPassword: isConfirming,
      password: '',
      currentDigit: 0
    );
  }

  // 비밀번호 저장 및 mm인증서 발급
  Future<bool> savePassword(String previousPassword) async {
    // 비밀번호 일치 확인
    if (state.password != previousPassword) {
      state = state.copyWith(
        error: '비밀번호가 일치하지 않습니다.',
        password: '',
        currentDigit: 0,
      );
      return false;
    }

    state = state.copyWith(isLoading: true);

    try {
      // 인증서 비밀번호 저장
      await _ref.read(certificateProvider.notifier).saveCertificatePassword(state.password);
      
      // 다음 단계로 진행 (인증서 발급은 별도 단계에서 처리)
      state = state.copyWith(isLoading: false);
      return true;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: '비밀번호 저장 중 오류가 발생했습니다: $e'
      );
      return false;
    }
  }
}

/*
  mm 인증서 로그인 상태 감지
*/
class MydataLoginNotifier extends StateNotifier<MydataLoginState> {
  final Ref _ref;

  MydataLoginNotifier(this._ref) : super(MydataLoginState());

  // 비밀번호 초기화
  void resetPassword() {
    state = state.copyWith(password: '', currentDigit: 0);
  }

  // 숫자 추가
  void addDigit(String digit) {
    if (state.currentDigit < 6) {
      final newPassword = state.password + digit;
      state = state.copyWith(
        password: newPassword,
        currentDigit: state.currentDigit + 1,
      );
    }
  }

  // mm 인증서 로그인 시도
  Future<bool> loginWithCertificate(String password) async {
    state = state.copyWith(isLoading: true, error: null);

    try {
      // 저장된 인증서 비밀번호 가져오기
      final savedPassword = await _ref.read(certificateProcessProvider.notifier).getSaveCertificatePassword();

      // 비밀번호 확인
      if (savedPassword == password) {
        // 로그인 성공
        state = state.copyWith(isLoading: false);
        return true;
      } else {
        // 로그인 실패
        state = state.copyWith(
          isLoading: false,
          error: '인증서 비밀번호 불일치',
        );
        return false;
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: '로그인 중 오류 발생: $e',
      );
      return false;
    }
  }
}

/*
  전자서명 동의 상태 감지
*/
class AgreementStateNotifier extends StateNotifier<AgreementState> {
  AgreementStateNotifier() : super(AgreementState());

  void toggleFirstAgreement() {
    state = state.copyWith(firstAgreement: !state.firstAgreement);
  }

  void toggleSecondAgreement() {
    state = state.copyWith(secondAgreement: !state.secondAgreement);
  }

  void setAtBottom(bool value) {
    state = state.copyWith(isAtBottom: value);
  }  
}

/*
  마이데이터 동의 상태 감지
*/
class MydataAgreementNotifier extends StateNotifier<MydataAgreementState> {
  final DigitalSignatureRepository _repository;

  MydataAgreementNotifier(this._repository) : super(MydataAgreementState());

  // 전자서명 검증 요청
  Future<bool> verifyDigitalSignature(String originalText) async {
    state = state.copyWith(isLoading: true, error: null);

    try {
      final response = await _repository.verifyDigitalSignature(originalText);
      
      final isVerified = response['data']['verified'] == true;
      final userKey = response['data']['userKey'] as String?;

      state = state.copyWith(
        isLoading: false,
        isCompleted: isVerified,
        isVerified: isVerified,
        userKey: userKey,
      );

      return isVerified;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: '전자서명 검증에 실패했습니다: $e',
      );
      return false;
    }
  }
}

/*
  mm 인증서 비밀번호 상태 프로바이더
*/
final MydataPasswordProvider = StateNotifierProvider<MydataPasswordNotifier, MydataPasswordState>((ref) {
  return MydataPasswordNotifier(ref);
});

/*
  mm 인증서 로그인 상태 프로바이더
*/
final MydataLoginProvider = StateNotifierProvider<MydataLoginNotifier, MydataLoginState>((ref) {
  return MydataLoginNotifier(ref);
});

/*
  전자서명 동의 상태 프로바이더
*/
final agreementStateProvider = StateNotifierProvider<AgreementStateNotifier, AgreementState>((ref) {
  return AgreementStateNotifier();
});

/*
  마이데이터 동의 상태 프로바이더
*/
final mydataAgreementProvider = StateNotifierProvider<MydataAgreementNotifier, MydataAgreementState>((ref) {
  final repository = ref.watch(digitalSignatureRepositoryProvider);
  return MydataAgreementNotifier(repository);
});