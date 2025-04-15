// 계좌 정보 모델
class AccountModel {
  final String bankCode;
  final String bankName;
  final String accountNo;
  final String accountName;

  AccountModel({
    required this.bankCode,
    required this.bankName,
    required this.accountNo,
    required this.accountName,
  });

  factory AccountModel.fromJson(Map<String, dynamic> json) {
    return AccountModel(
      bankCode: json['bankCode'] as String,
      bankName: json['bankName'] as String,
      accountNo: json['accountNo'] as String,
      accountName: json['accountName'] as String,
    );
  }
}

// 계좌 목록 응답 모델
class AccountListResponse {
  final int code;
  final String message;
  final AccountListData? data;

  AccountListResponse({
    required this.code,
    required this.message,
    this.data,
  });

  factory AccountListResponse.fromJson(Map<String, dynamic> json) {
    return AccountListResponse(
      code: json['code'] as int,
      message: json['message'] as String,
      data: json['data'] != null
          ? AccountListData.fromJson(json['data'])
          : null,
    );
  }
}

class AccountListData {
  final List<AccountModel> accountList;

  AccountListData({required this.accountList});

  factory AccountListData.fromJson(Map<String, dynamic> json) {
    return AccountListData(
      accountList: (json['accountList'] as List)
          .map((item) => AccountModel.fromJson(item))
          .toList(),
    );
  }
}

// 입금 내역 모델
class DepositModel {
  final String transactionDate;
  final String transactionTime;
  final int transactionBalance;
  final String transactionSummary;
  final String transactionMemo;

  DepositModel({
    required this.transactionDate,
    required this.transactionTime,
    required this.transactionBalance,
    required this.transactionSummary,
    required this.transactionMemo,
  });

  factory DepositModel.fromJson(Map<String, dynamic> json) {
    return DepositModel(
      transactionDate: json['transactionDate'] as String,
      transactionTime: json['transactionTime'] as String,
      transactionBalance: json['transactionBalance'] as int,
      transactionSummary: json['transactionSummary'] as String,
      transactionMemo: json['transactionMemo'] as String,
    );
  }
}

// 입금 내역 응답 모델
class DepositListResponse {
  final int code;
  final String message;
  final DepositListData? data;

  DepositListResponse({
    required this.code,
    required this.message,
    this.data,
  });

  factory DepositListResponse.fromJson(Map<String, dynamic> json) {
    return DepositListResponse(
      code: json['code'] as int,
      message: json['message'] as String,
      data: json['data'] != null
          ? DepositListData.fromJson(json['data'])
          : null,
    );
  }
}

class DepositListData {
  final List<DepositModel> depositList;

  DepositListData({required this.depositList});

  factory DepositListData.fromJson(Map<String, dynamic> json) {
    return DepositListData(
      depositList: (json['depositList'] as List)
          .map((item) => DepositModel.fromJson(item))
          .toList(),
    );
  }
}

// 월급 정보 모델
class SalaryModel {
  final int salary;
  final int date;

  SalaryModel({
    required this.salary,
    required this.date,
  });

  factory SalaryModel.fromJson(Map<String, dynamic> json) {
    return SalaryModel(
      salary: json['salary'] as int,
      date: json['date'] as int,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'salary': salary,
      'date': date,
    };
  }
}

// 월급 등록/수정 응답 모델
class SalaryResponse {
  final int code;
  final String message;
  final SalaryResponseData? data;

  SalaryResponse({
    required this.code,
    required this.message,
    this.data,
  });

  factory SalaryResponse.fromJson(Map<String, dynamic> json) {
    return SalaryResponse(
      code: json['code'] as int,
      message: json['message'] as String,
      data: json['data'] != null
          ? SalaryResponseData.fromJson(json['data'])
          : null,
    );
  }
}

class SalaryResponseData {
  final String message;

  SalaryResponseData({required this.message});

  factory SalaryResponseData.fromJson(Map<String, dynamic> json) {
    return SalaryResponseData(
      message: json['message'] as String,
    );
  }
}