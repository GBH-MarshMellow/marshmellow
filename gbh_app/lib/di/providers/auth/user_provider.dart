import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/data/models/auth/user_state.dart';
import 'package:marshmellow/di/providers/core_providers.dart';

class UserStateNotifier extends StateNotifier<UserState> {
  final FlutterSecureStorage _secureStorage;

  UserStateNotifier(this._secureStorage) : super(UserState()) {
    // 초기화시 저장된 값이 있는지 확인
    _loadUserData();
  }

  // 본인확인 정보 저장
  Future<void> setVerificationData({
    required String userName,
    required String phoneNumber,
    required String userCode,
    required String carrier,
  }) async {
    // 상태 업데이트
    state = state.copyWith(
      userName: userName,
      phoneNumber: phoneNumber,
      userCode: userCode,
      carrier: carrier,
    );

    // secure storage에 저장
    await _secureStorage.write(key: StorageKeys.userName, value: userName);
    await _secureStorage.write(key: StorageKeys.phoneNumber, value: phoneNumber);
    await _secureStorage.write(key: StorageKeys.userCode, value: userCode);
    await _secureStorage.write(key: StorageKeys.carrier, value: carrier);
  }

  // 사용자 인증 상태 설정(회원가입/로그인 성공시)
  Future<void> setAuthenticated(bool authenticated) async {
    state = state.copyWith(isAuthenticated: authenticated);
  }

  // 저장된 데이터 가져오기
  Future<void> _loadUserData() async {
    final userName = await _secureStorage.read(key: StorageKeys.userName);
    final phoneNumber = await _secureStorage.read(key: StorageKeys.phoneNumber);
    final userCode = await _secureStorage.read(key: StorageKeys.userCode);
    final carrier = await _secureStorage.read(key: StorageKeys.carrier);

    if (userName != null && phoneNumber != null && userCode != null) {
      state = state.copyWith(
        userName: userName,
        phoneNumber: phoneNumber,
        userCode: userCode,
        carrier: carrier,
      );
    }
  }
}

// provider 정의
final userStateProvider = StateNotifierProvider<UserStateNotifier, UserState>((ref) {
  final secureStorage = ref.watch(secureStorageProvider);
  return UserStateNotifier(secureStorage);
});