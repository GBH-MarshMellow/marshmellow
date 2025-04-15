import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/data/datasources/remote/auth_api.dart';
import 'package:marshmellow/data/models/auth/user_state.dart';

/*
  핀번호를 사용한 회원가입 repository
*/
class AuthRepository {
  final AuthApi _authApi;
  final FlutterSecureStorage _secureStorage;

  AuthRepository(this._authApi, this._secureStorage);

  // 회원가입
  Future<bool> signUp({
    required String userName,
    required String phoneNumber,
    required String userCode,
    required String pin,
    required String fcmToken
  }) async {
    print('회원가입 시도');
    print('userName: $userName');
    print('phoneNumber: $phoneNumber');
    print('userCode: $userCode');
    print('pin: $pin');
    print('fcm: $fcmToken');

    try {
      final response = await _authApi.signUp(
        userName: userName, 
        phoneNumber: phoneNumber, 
        userCode: userCode, 
        pin: pin,
        fcmToken: fcmToken
      );

      print('API 응답: $response');

      if (response['code'] == 200) {
        // 토큰 저장
        await _saveTokens(
          response['data']['accessToken'],
          response['data']['refreshToken'],
        );
        print('토큰 저장 성공');
        return true;
      }
      print('회원가입 실패: 응답 코드 불일치');
      return false;
    } catch (e) {
      print('회원가입 오류: $e');
      return false;
    }
  }

  // PIN 로그인
  Future<bool> loginWithPin({required String phoneNumber, required String pin}) async {
    try {
      final response = await _authApi.loginWithPin(
        phoneNumber: phoneNumber, 
        pin: pin
      );

      if (response['code'] == 200) {
        // 토큰 저장
        await _saveTokens(
          response['data']['accessToken'],
          response['data']['refreshToken'], 
        );
        return true;
      }
      return false;
    } catch (e) {
      print('PIN 로그인 오류: $e');
      return false;
    }
  }

  // 생체인식 로그인
  Future<bool> loginWithBiometrics({required String phoneNumber}) async {
    try {
      final response = await _authApi.loginWithBiometrics(
        phoneNumber: phoneNumber
      );

      if (response['code'] == 200) {
        // 토큰 저장
        await _saveTokens(
          response['data']['accessToken'],
          response['data']['refreshToken'], 
        );
        return true;
      }
      return false;
    } catch (e) {
      print('생체인식 로그인 오류: $e');
      return false;
    }
  }

  // 토큰 갱신
  Future<bool> reissueToken() async {
    try {
      final currentRefreshToken = await _secureStorage.read(key: StorageKeys.refreshToken);

      if (currentRefreshToken == null) {
        return false;
      }

      final response = await _authApi.reissueToken(refreshToken: currentRefreshToken);

      if (response['code'] == 200) {
        // 토큰 업데이트
        await _saveTokens(
          response['data']['accessToken'],
          response['data']['refreshToken'], 
        );
        return true;
      }
      return false;
    } catch (e) {
      print('토큰 갱신 오류: $e');
      return false;
    }
  }

  // 로그아웃
  Future<bool> logout() async {
    try {
      final response = await _authApi.logout();

      if (response['code'] == 200) {
        // 저장된 토큰 삭제
        await _secureStorage.delete(key: StorageKeys.accessToken);
        await _secureStorage.delete(key: StorageKeys.refreshToken);
        return true;
      }
      return false;
    } catch (e) {
      print('로그아웃 오류: $e');
      return false;
    }
  }

  // 토큰 저장 메서드
  Future<void> _saveTokens(String accessToken, String refreshToken) async {
    await _secureStorage.write(key: StorageKeys.accessToken, value: accessToken);
    await _secureStorage.write(key: StorageKeys.refreshToken, value: refreshToken);
  }

  // 생체인식 설정 저장
  Future<void> saveBiometricPreference(bool useBiometrics) async {
    await _secureStorage.write(key: StorageKeys.useBiometrics, value: useBiometrics.toString());
  }

  // 생체인식 설정 불러오기
  Future<bool> getBiometricPreference() async {
    final value = await _secureStorage.read(key: StorageKeys.useBiometrics);
    return value == 'true';
  }

  // 사용자 전화번호 저장
  Future<void> saveUserPhoneNumber(String phoneNumber) async {
    await _secureStorage.write(key: StorageKeys.phoneNumber, value: phoneNumber);
  }

  // 사용자 전화번호 불러오기
  Future<String?> getUserPhoneNumber() async {
    return await _secureStorage.read(key: StorageKeys.phoneNumber);
  }
}