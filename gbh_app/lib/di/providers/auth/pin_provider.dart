import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:local_auth/local_auth.dart';
import 'package:marshmellow/data/models/auth/pin_state.dart';
import 'package:marshmellow/data/repositories/auth/auth_repository.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'package:marshmellow/di/providers/auth/user_provider.dart';
import 'package:marshmellow/di/providers/core_providers.dart';
import 'package:shared_preferences/shared_preferences.dart';

// 이전 PIN 임시 저장 프로바이더
final previousPinProvider = StateProvider<String>((ref) => '');

// 회원 repository 프로바이더
final authRepositoryProvider = Provider<AuthRepository>((ref) {
  final authApi = ref.watch(authApiProvider);
  final secureStorage = ref.watch(secureStorageProvider);
  return AuthRepository(authApi, secureStorage);
});

// PIN 상태 프로바이더
class PinStateNotifier extends StateNotifier<PinState> {
  final AuthRepository _authRepository;
  final LocalAuthentication _localAuth = LocalAuthentication();
  final Ref _ref;

  PinStateNotifier(this._authRepository, this._ref) : super(PinState()) {
    _checkBiometrics();
  }

  // 생체인식 가능 여부 확인
  Future<void> _checkBiometrics() async {
    try {
      final canCheckBiometrics = await _localAuth.canCheckBiometrics;
      final isDeviceSupported = await _localAuth.isDeviceSupported();
      final isBiometricsAvailable = canCheckBiometrics && isDeviceSupported;

      state = state.copyWith(isBiometricsAvailable: isBiometricsAvailable);
    } on PlatformException catch (e) {
      print('생체인식 확인 실패: $e');
    }
  }

  // PIN 한자리 추가
  void addDigit(String digit) {
    if (state.currentDigit < 4) {
      final newPin = state.pin + digit;
      state = state.copyWith(
        pin: newPin,
        currentDigit: state.currentDigit + 1,
      );
    }
  }

  // PIN 한자리 삭제
  void removeDigit() {
    if (state.currentDigit > 0) {
      final newPin = state.pin.substring(0, state.pin.length -1);
      state = state.copyWith(
        pin: newPin,
        currentDigit: state.currentDigit -1,
      );
    }
  }

  // PIN 초기화
  void resetPin() {
    state = state.copyWith(
      pin: '',
      currentDigit: 0,
    );
  }

  // 확인 모드 설정
  void setConfirmMode(bool isConfirming) {
    resetPin();
    state = state.copyWith(isConfirmingPin: isConfirming);
  }

  // 생체인식 사용 여부 설정
  void toggleBiometrics(bool useBiometrics) async {
    state = state.copyWith(useBiometrics: useBiometrics);
    await _authRepository.saveBiometricPreference(useBiometrics);
  }

  // PIN 저장 및 회원가입 API 호출
  Future<bool> savePin(String previousPin) async {
    final FirebaseMessaging _firebaseMessaging = FirebaseMessaging.instance;
    final String? fcmToken = await _firebaseMessaging.getToken();

    if (fcmToken == null) {
      print('FCM token is null, using empty string');
    }

    // PIN 일치 검사
    if (state.pin != previousPin) {
      resetPin();
      setConfirmMode(false);
      return false;
    }

    // 사용자 정보 가져오기
    final userInfo = await _getUserInfo();
    if (userInfo.isEmpty) {
      return false;
    }

    // userCode 형변환
    String userCode = userInfo['userCode'] ?? '';

    if (!userCode.contains('-')) {
      // 주민번호 앞 6자리와 마지막 숫자 사이에 - 추가
      userCode = '${userCode.substring(0,6)}-${userCode.substring(6,7)}';
    }

    // 회원가입 api 호출
    final success = await _authRepository.signUp(
      userName: userInfo['userName'] ?? '',
      phoneNumber: userInfo['phoneNumber'] ?? '',
      userCode: userCode,
      pin: state.pin,
      fcmToken: fcmToken ?? '',
    );

    if (success) {
      // 전화번호 저장
      await _authRepository.saveUserPhoneNumber(userInfo['phoneNumber'] ?? '');

      // 생체인식 설정 저장
      if (state.useBiometrics) {
        await _authRepository.saveBiometricPreference(true);
      }

      resetPin();
      return true;
    }

    return false;
  }
  
  // 사용자 정보 가져오기 
  Future<Map<String, String>> _getUserInfo() async {
    final userState = _ref.read(userStateProvider);

    print('사용자 정보 확인');
    print('userName: ${userState.userName}');
    print('phoneNumber: ${userState.phoneNumber}');
    print('userCode: ${userState.userCode}');

    if (userState.userName != null && userState.phoneNumber != null && userState.userCode != null) {
      return {
      'userName': userState.userName!,
      'phoneNumber': userState.phoneNumber!,
      'userCode': userState.userCode!,
      };
    }
    print('사용자 정보 없음');
    // 정보 없으면 빈 맵 반환
    return {};
  }

  // PIN 로그인
  Future<bool> loginWithPin() async {
    final userState = _ref.read(userStateProvider);
    final phoneNumber = userState.phoneNumber;

    if (phoneNumber == null) return false;

    return await _authRepository.loginWithPin(
      phoneNumber: phoneNumber, 
      pin: state.pin
    );
  }

  // 생체인식 로그인
  Future<bool> loginWithBiometrics() async {
    final userState = _ref.read(userStateProvider);
    final phoneNumber = userState.phoneNumber;

    if (phoneNumber == null) return false;

    try {
      // 생체인식 인증
      final authenticated = await _localAuth.authenticate(
        localizedReason: '생체인식으로 로그인',
        options: const AuthenticationOptions(
          stickyAuth: true,
          biometricOnly: true,
        )
      );

      if (authenticated) {
        // 서버에 생체인식 로그인 요청
        return await _authRepository.loginWithBiometrics(phoneNumber: phoneNumber);
      }

      return false;
    } catch (e) {
      print('생체인식 로그인 오류: $e');
      return false;
    }
  }
}

// provider
final pinStateProvider = StateNotifierProvider<PinStateNotifier, PinState>((ref) {
  final authRepository = ref.watch(authRepositoryProvider);
  return PinStateNotifier(authRepository, ref);
});