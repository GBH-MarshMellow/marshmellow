// lib/presentation/viewmodels/my/user_info_viewmodel.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/api_client.dart';
import 'package:marshmellow/data/datasources/remote/my/salary_api.dart';
import 'package:marshmellow/data/models/my/user_detail_info.dart';
import 'package:marshmellow/di/providers/api_providers.dart';

// API Provider
final mySalaryApiProvider = Provider<MySalaryApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return MySalaryApi(apiClient);
});

// State 클래스
class UserInfoState {
  final UserDetailInfo userDetail;
  final bool isLoading;
  final String? error;

  UserInfoState({
    required this.userDetail,
    this.isLoading = false,
    this.error,
  });

  // 초기 상태
  factory UserInfoState.initial() {
    return UserInfoState(
      userDetail: UserDetailInfo.empty(),
      isLoading: false,
      error: null,
    );
  }

  // 상태 업데이트 메서드
  UserInfoState copyWith({
    UserDetailInfo? userDetail,
    bool? isLoading,
    String? error,
  }) {
    return UserInfoState(
      userDetail: userDetail ?? this.userDetail,
      isLoading: isLoading ?? this.isLoading,
      error: error ?? this.error,
    );
  }
}

// NotifierProvider
class UserInfoNotifier extends StateNotifier<UserInfoState> {
  final MySalaryApi _mySalaryApi;

  UserInfoNotifier(this._mySalaryApi) : super(UserInfoState.initial()) {
    loadAllUserInfo();
  }

  Future<void> loadAllUserInfo() async {
    await loadUserDetailFromApi();
  }

  Future<void> loadUserDetailFromApi() async {
    try {
      state = state.copyWith(isLoading: true, error: null);
      
final data = await _mySalaryApi.getMyInfoDetail();
// 안전한 타입 확인 후 처리
if (data is Map<String, dynamic>) {
  final userDetail = UserDetailInfo.fromJson(data);
  state = state.copyWith(userDetail: userDetail, isLoading: false);
} else {
  state = state.copyWith(
    isLoading: false,
    error: "API 응답 형식이 올바르지 않습니다."
  );
}      // data는 이미 응답의 'data' 부분만 들어있으므로 바로 사용
      final userDetail = UserDetailInfo.fromJson(data as Map<String, dynamic>);
      
      state = state.copyWith(
        userDetail: userDetail, 
        isLoading: false
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: "API 데이터 로드 실패: ${e.toString()}"
      );
    }
  }

  // 월급액 수정 메서드 (필요 시 사용)
  Future<void> updateSalary(int salary, int date) async {
    try {
      state = state.copyWith(isLoading: true, error: null);
      await _mySalaryApi.updateSalary(salary, date);
      // 업데이트 후 최신 데이터 다시 로드
      await loadUserDetailFromApi();
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: "월급 정보 업데이트 실패: ${e.toString()}"
      );
    }
  }
  // 월급 등록 메서드
  Future<void> myRegisterSalary(int salary, int date, String account) async {
    try {
      state = state.copyWith(isLoading: true, error: null);
      await _mySalaryApi.myRegisterSalary(salary, date, account);
      // 등록 후 최신 데이터 다시 로드
      await loadUserDetailFromApi();
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: "월급 정보 등록 실패: ${e.toString()}"
      );
    }
  }

  // 월급액 수정 메서드 (기존 메서드 수정)
  Future<void> myUpdateSalary(int salary, int date, String account) async {
    try {
      state = state.copyWith(isLoading: true, error: null);
      await _mySalaryApi.myUpdateSalary(salary, date, account);
      // 업데이트 후 최신 데이터 다시 로드
      await loadUserDetailFromApi();
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: "월급 정보 업데이트 실패: ${e.toString()}"
      );
    }
  }
}




// Provider 정의 부분
final userInfoProvider = StateNotifierProvider<UserInfoNotifier, UserInfoState>((ref) {
  final mySalaryApi = ref.watch(mySalaryApiProvider);
  return UserInfoNotifier(mySalaryApi);
});