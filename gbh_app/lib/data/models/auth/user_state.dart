/*
  사용자 정보 모델
*/
class UserState {
  final String? userName;
  final String? phoneNumber;
  final String? userCode;
  final String? carrier;
  final bool isAuthenticated;

  UserState({
    this.userName,
    this.phoneNumber,
    this.userCode,
    this.carrier,
    this.isAuthenticated = false,
  });

  UserState copyWith({
    String? userName,
    String? phoneNumber,
    String? userCode,
    String? carrier,
    bool? isAuthenticated,
  }) {
    return UserState(
      userName: userName ?? this.userName,
      phoneNumber: phoneNumber ?? this.phoneNumber,
      userCode: userCode ?? this.userCode,
      carrier: carrier ?? this.carrier,
      isAuthenticated: isAuthenticated ?? this.isAuthenticated,
    );
  }
}