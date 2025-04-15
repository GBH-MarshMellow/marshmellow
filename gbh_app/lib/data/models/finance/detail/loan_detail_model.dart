// lib/data/models/finance/loan_detail_model.dart

class LoanDetailResponse {
  final int code;
  final String message;
  final LoanDetailData data;

  LoanDetailResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory LoanDetailResponse.fromJson(Map<String, dynamic> json) {
    return LoanDetailResponse(
      code: json['code'],
      message: json['message'],
      data: LoanDetailData.fromJson(json['data']),
    );
  }
}

class LoanDetailData {
  // IV 필드 추가
  final String iv;
  final String status;  // 암호화된 상태값
  final String loanBalance;  // int에서 String으로 변경 (암호화된 값)
  final String remainingLoanBalance;  // int에서 String으로 변경 (암호화된 값)
  final List<RepaymentRecord> repaymentRecords;

  LoanDetailData({
    required this.iv,
    required this.status,
    required this.loanBalance,
    required this.remainingLoanBalance,
    required this.repaymentRecords,
  });

  factory LoanDetailData.fromJson(Map<String, dynamic> json) {
    return LoanDetailData(
      iv: json['iv'],
      status: json['status'],
      loanBalance: json['loanBalance'],
      remainingLoanBalance: json['remainingLoanBalance'],
      repaymentRecords: (json['repaymentRecords'] as List)
          .map((item) => RepaymentRecord.fromJson(item))
          .toList(),
    );
  }
}

class RepaymentRecord {
  final String installmentNumber;
  final String repaymentAttemptTime;
  final String repaymentActualTime;  // nullable 제거 (항상 값이 오는 듯)
  final String repaymentAttemptDate;
  final String failureReason;
  final String paymentBalance;
  final String repaymentActualDate;  // nullable 제거 (항상 값이 오는 듯)
  final String status;

  RepaymentRecord({
    required this.installmentNumber,
    required this.repaymentAttemptTime,
    required this.repaymentActualTime,
    required this.repaymentAttemptDate,
    required this.failureReason,
    required this.paymentBalance,
    required this.repaymentActualDate,
    required this.status,
  });

  factory RepaymentRecord.fromJson(Map<String, dynamic> json) {
    return RepaymentRecord(
      installmentNumber: json['installmentNumber'],
      repaymentAttemptTime: json['repaymentAttemptTime'],
      repaymentActualTime: json['repaymentActualTime'],
      repaymentAttemptDate: json['repaymentAttemptDate'],
      failureReason: json['failureReason'],
      paymentBalance: json['paymentBalance'],
      repaymentActualDate: json['repaymentActualDate'],
      status: json['status'],
    );
  }
}