// presentation/viewmodels/my/user_secure_info_viewmodel.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/di/providers/core_providers.dart';

class UserSecureInfoState {
  final String? userName;
  final String? phoneNumber;
  final String? certificateStatus;
  final String? certificateEmail;
  final bool isLoading;
  final String? error;

  UserSecureInfoState({
    this.userName,
    this.phoneNumber,
    this.certificateStatus,
    this.certificateEmail,
    this.isLoading = false,
    this.error,
  });

  UserSecureInfoState copyWith({
    String? userName,
    String? phoneNumber,
    String? certificateStatus,
    String? certificateEmail,
    bool? isLoading,
    String? error,
  }) {
    return UserSecureInfoState(
      userName: userName ?? this.userName,
      phoneNumber: phoneNumber ?? this.phoneNumber,
      certificateStatus: certificateStatus ?? this.certificateStatus,
      certificateEmail: certificateEmail ?? this.certificateEmail,
      isLoading: isLoading ?? this.isLoading,
      error: error ?? this.error,
    );
  }
}

class UserSecureInfoNotifier extends StateNotifier<UserSecureInfoState> {
  final FlutterSecureStorage _secureStorage;

  UserSecureInfoNotifier(this._secureStorage) : super(UserSecureInfoState(isLoading: true)) {
    loadUserInfo();
  }

  Future<void> loadUserInfo() async {
    try {
      state = state.copyWith(isLoading: true, error: null);
      
      final userName = await _secureStorage.read(key: StorageKeys.userName);
      final phoneNumber = await _secureStorage.read(key: StorageKeys.phoneNumber);
      final certificateStatus = await _secureStorage.read(key: StorageKeys.certificateStatus);
      final certificateEmail = await _secureStorage.read(key: StorageKeys.certificateEmail);
      
      state = state.copyWith(
        userName: userName,
        phoneNumber: phoneNumber,
        certificateStatus: certificateStatus,
        certificateEmail: certificateEmail,
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false, 
        error: '사용자 정보를 불러오는 중 오류가 발생했습니다: $e'
      );
    }
  }
}

// 프로바이더 정의
final userSecureInfoProvider = StateNotifierProvider<UserSecureInfoNotifier, UserSecureInfoState>((ref) {
  final secureStorage = ref.watch(secureStorageProvider);
  return UserSecureInfoNotifier(secureStorage);
});