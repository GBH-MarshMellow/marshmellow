// withdrawal_account_model.dart

// 출금계좌 목록 조회 응답 모델
class WithdrawalAccountResponse {
  final int code;
  final String message;
  final WithdrawalAccountData data;

  WithdrawalAccountResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory WithdrawalAccountResponse.fromJson(Map<String, dynamic> json) {
    return WithdrawalAccountResponse(
      code: json['code'],
      message: json['message'],
      data: WithdrawalAccountData.fromJson(json['data']),
    );
  }
}

// 출금계좌 데이터 모델
class WithdrawalAccountData {
  final List<WithdrawalAccountItem> withdrawalAccountList;

  WithdrawalAccountData({
    required this.withdrawalAccountList,
  });

  factory WithdrawalAccountData.fromJson(Map<String, dynamic> json) {
    return WithdrawalAccountData(
      withdrawalAccountList: (json['withdrawalAccountList'] as List)
          .map((item) => WithdrawalAccountItem.fromJson(item))
          .toList(),
    );
  }
}

// 출금계좌 항목 모델
class WithdrawalAccountItem {
  final int withdrawalAccountId;
  final String accountNo;

  WithdrawalAccountItem({
    required this.withdrawalAccountId,
    required this.accountNo,
  });

  factory WithdrawalAccountItem.fromJson(Map<String, dynamic> json) {
    return WithdrawalAccountItem(
      withdrawalAccountId: json['withdrawalAccountId'],
      accountNo: json['accountNo'],
    );
  }
}

// 계좌 인증 발송 응답 모델
class AccountAuthResponse {
  final int code;
  final String message;
  final AccountAuthData data;

  AccountAuthResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory AccountAuthResponse.fromJson(Map<String, dynamic> json) {
    return AccountAuthResponse(
      code: json['code'],
      message: json['message'],
      data: AccountAuthData.fromJson(json['data']),
    );
  }
}

// 계좌 인증 데이터 모델
class AccountAuthData {
  final String iv;
  final String authCode;

  AccountAuthData({
    required this.iv,
    required this.authCode,
  });

  factory AccountAuthData.fromJson(Map<String, dynamic> json) {
    return AccountAuthData(
      iv: json['iv'],
      authCode: json['authCode'],
    );
  }
}

// 계좌 인증 검증 응답 모델
class AccountAuthVerifyResponse {
  final int code;
  final String message;
  final AccountAuthVerifyData data;

  AccountAuthVerifyResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory AccountAuthVerifyResponse.fromJson(Map<String, dynamic> json) {
    return AccountAuthVerifyResponse(
      code: json['code'],
      message: json['message'],
      data: AccountAuthVerifyData.fromJson(json['data']),
    );
  }
}

// 계좌 인증 검증 데이터 모델
class AccountAuthVerifyData {
  final String status;
  final int withdrawalAccountId;

  AccountAuthVerifyData({
    required this.status,
    required this.withdrawalAccountId,
  });

  factory AccountAuthVerifyData.fromJson(Map<String, dynamic> json) {
    return AccountAuthVerifyData(
      status: json['status'],
      withdrawalAccountId: json['withdrawalAccountId'],
    );
  }
}