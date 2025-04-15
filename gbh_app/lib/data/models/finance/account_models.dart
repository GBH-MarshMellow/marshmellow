// 입출금 계좌 모델
class DemandDepositData {
  //**********************************************
  //* 변경: totalAmount를 String 타입으로 변경 (암호화된 값이므로)
  //**********************************************
  final String totalAmount;  // int에서 String으로 변경
  final List<DemandDepositItem> demandDepositList;

  DemandDepositData({
    required this.totalAmount,
    required this.demandDepositList,
  });

  factory DemandDepositData.fromJson(Map<String, dynamic> json) {
    return DemandDepositData(
      //**********************************************
      //* 변경: 원래 정수형이었던 totalAmount가 이제는 암호화된 문자열
      //**********************************************
      totalAmount: json['totalAmount'],
      demandDepositList: (json['demandDepositList'] as List)
          .map((item) => DemandDepositItem.fromJson(item))
          .toList(),
    );
  }
}

class DemandDepositItem {
  final String bankCode;
  final String bankName;
  final String accountNo;
  final String accountName;
  //**********************************************
  //* 변경: accountBalance를 String 타입으로 변경하고 nullable로 설정
  //* 추가: encodedAccountBalance 필드 추가
  //**********************************************
  final int? accountBalance;  // nullable로 변경
  final String? encodedAccountBalance;  // 암호화된 잔액 필드 추가

  DemandDepositItem({
    required this.bankCode,
    required this.bankName,
    required this.accountNo,
    required this.accountName,
    this.accountBalance,  // nullable로 변경
    this.encodedAccountBalance,  // 새 필드 추가
  });

  factory DemandDepositItem.fromJson(Map<String, dynamic> json) {
    return DemandDepositItem(
      bankCode: json['bankCode'],
      bankName: json['bankName'],
      accountNo: json['accountNo'],
      accountName: json['accountName'],
      //**********************************************
      //* 변경: accountBalance는 이제 사용하지 않고, encodedAccountBalance를 사용
      //**********************************************
      accountBalance: json['accountBalance'],
      encodedAccountBalance: json['encodedAccountBalance'],
    );
  }
}

// 대출 모델
class LoanData {
  //**********************************************
  //* 변경: totalAmount를 String 타입으로 변경 (암호화된 값이므로)
  //**********************************************
  final String totalAmount;  // int에서 String으로 변경
  final List<LoanItem> loanList;

  LoanData({
    required this.totalAmount,
    required this.loanList,
  });

  factory LoanData.fromJson(Map<String, dynamic> json) {
    return LoanData(
      //**********************************************
      //* 변경: 원래 정수형이었던 totalAmount가 이제는 암호화된 문자열
      //**********************************************
      totalAmount: json['totalAmount'],
      loanList: (json['loanList'] as List)
          .map((item) => LoanItem.fromJson(item))
          .toList(),
    );
  }
}

class LoanItem {
  final String accountNo;
  final String accountName;
  //**********************************************
  //* 변경: loanBalance를 nullable로 변경
  //* 추가: encodeLoanBalance 필드 추가
  //**********************************************
  final int? loanBalance;  // nullable로 변경
  final String? encodeLoanBalance;  // 암호화된 대출 잔액 필드 추가

  LoanItem({
    required this.accountNo,
    required this.accountName,
    this.loanBalance,  // nullable로 변경
    this.encodeLoanBalance,  // 새 필드 추가
  });

  factory LoanItem.fromJson(Map<String, dynamic> json) {
    return LoanItem(
      accountNo: json['accountNo'],
      accountName: json['accountName'],
      //**********************************************
      //* 변경: loanBalance는 이제 사용하지 않고, encodeLoanBalance를 사용
      //**********************************************
      loanBalance: json['loanBalance'],
      encodeLoanBalance: json['encodeLoanBalance'],
    );
  }
}

// 적금 모델
class SavingsData {
  //**********************************************
  //* 변경: totalAmount를 String 타입으로 변경 (암호화된 값이므로)
  //**********************************************
  final String totalAmount;  // int에서 String으로 변경
  final List<SavingsItem> savingsList;

  SavingsData({
    required this.totalAmount,
    required this.savingsList,
  });

  factory SavingsData.fromJson(Map<String, dynamic> json) {
    return SavingsData(
      //**********************************************
      //* 변경: 원래 정수형이었던 totalAmount가 이제는 암호화된 문자열
      //**********************************************
      totalAmount: json['totalAmount'],
      savingsList: (json['savingsList'] as List)
          .map((item) => SavingsItem.fromJson(item))
          .toList(),
    );
  }
}

class SavingsItem {
  final String bankCode;
  final String bankName;
  final String accountNo;
  final String accountName;
  //**********************************************
  //* 변경: totalBalance를 nullable로 변경
  //* 추가: encodedTotalBalance 필드 추가
  //**********************************************
  final int? totalBalance;  // nullable로 변경
  final String? encodedTotalBalance;  // 암호화된 잔액 필드 추가

  SavingsItem({
    required this.bankCode,
    required this.bankName,
    required this.accountNo,
    required this.accountName,
    this.totalBalance,  // nullable로 변경
    this.encodedTotalBalance,  // 새 필드 추가
  });

  factory SavingsItem.fromJson(Map<String, dynamic> json) {
    return SavingsItem(
      bankCode: json['bankCode'],
      bankName: json['bankName'],
      accountNo: json['accountNo'],
      accountName: json['accountName'],
      //**********************************************
      //* 변경: totalBalance는 이제 사용하지 않고, encodedTotalBalance를 사용
      //**********************************************
      totalBalance: json['totalBalance'],
      encodedTotalBalance: json['encodedTotalBalance'],
    );
  }
}

// 예금 모델
class DepositData {
  //**********************************************
  //* 변경: totalAmount를 String 타입으로 변경 (암호화된 값이므로)
  //**********************************************
  final String totalAmount;  // int에서 String으로 변경
  final List<DepositItem> depositList;

  DepositData({
    required this.totalAmount,
    required this.depositList,
  });

  factory DepositData.fromJson(Map<String, dynamic> json) {
    return DepositData(
      //**********************************************
      //* 변경: 원래 정수형이었던 totalAmount가 이제는 암호화된 문자열
      //**********************************************
      totalAmount: json['totalAmount'],
      depositList: (json['depositList'] as List)
          .map((item) => DepositItem.fromJson(item))
          .toList(),
    );
  }
}

class DepositItem {
  final String bankCode;
  final String bankName;
  final String accountNo;
  final String accountName;
  //**********************************************
  //* 변경: depositBalance를 nullable로 변경
  //* 추가: encodeDepositBalance 필드 추가
  //**********************************************
  final int? depositBalance;  // nullable로 변경
  final String? encodeDepositBalance;  // 암호화된 잔액 필드 추가

  DepositItem({
    required this.bankCode,
    required this.bankName,
    required this.accountNo,
    required this.accountName,
    this.depositBalance,  // nullable로 변경
    this.encodeDepositBalance,  // 새 필드 추가
  });

  factory DepositItem.fromJson(Map<String, dynamic> json) {
    return DepositItem(
      bankCode: json['bankCode'],
      bankName: json['bankName'],
      accountNo: json['accountNo'],
      accountName: json['accountName'],
      //**********************************************
      //* 변경: depositBalance는 이제 사용하지 않고, encodeDepositBalance를 사용
      //**********************************************
      depositBalance: json['depositBalance'],
      encodeDepositBalance: json['encodeDepositBalance'],
    );
  }
}