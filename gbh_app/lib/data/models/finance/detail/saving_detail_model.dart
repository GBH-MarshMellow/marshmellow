// lib/data/models/finance/saving_detail_model.dart

class SavingDetailResponse {
  final int code;
  final String message;
  final SavingDetailData data;

  SavingDetailResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory SavingDetailResponse.fromJson(Map<String, dynamic> json) {
    return SavingDetailResponse(
      code: json['code'],
      message: json['message'],
      data: SavingDetailData.fromJson(json['data']),
    );
  }
}

class SavingDetailData {
  final String iv;  // IV 필드 추가
  final SavingPaymentItem paymentList;  // 리스트가 아닌 단일 객체로 변경

  SavingDetailData({
    required this.iv,
    required this.paymentList,
  });

  factory SavingDetailData.fromJson(Map<String, dynamic> json) {
    return SavingDetailData(
      iv: json['iv'],
      paymentList: SavingPaymentItem.fromJson(json['paymentList']),  // 단일 객체 파싱
    );
  }
}

class SavingPaymentItem {
  final String bankCode;
  final String bankName;
  final String accountNo;
  final String? accountName;  // nullable로 변경 또는 제거
  final String interestRate;
  final String depositBalance;
  final String totalBalance;
  final String accountCreateDate;
  final String accountExpiryDate;
  final List<PaymentInfoItem> paymentInfo;

  SavingPaymentItem({
    required this.bankCode,
    required this.bankName,
    required this.accountNo,
    this.accountName,  // 선택적으로 변경
    required this.interestRate,
    required this.depositBalance,
    required this.totalBalance,
    required this.accountCreateDate,
    required this.accountExpiryDate,
    required this.paymentInfo,
  });

  factory SavingPaymentItem.fromJson(Map<String, dynamic> json) {
    return SavingPaymentItem(
      bankCode: json['bankCode'],
      bankName: json['bankName'],
      accountNo: json['accountNo'],
      accountName: json['accountName'],  // null이면 null로 설정됨
      interestRate: json['interestRate'],
      depositBalance: json['depositBalance'],
      totalBalance: json['totalBalance'],
      accountCreateDate: json['accountCreateDate'],
      accountExpiryDate: json['accountExpiryDate'],
      paymentInfo: (json['paymentInfo'] as List)
          .map((item) => PaymentInfoItem.fromJson(item))
          .toList(),
    );
  }
}

class PaymentInfoItem {
  final String depositInstallment;
  final String paymentBalance;
  final String paymentDate;
  final String paymentTime;
  final String status;
  final String? failureReason;

  PaymentInfoItem({
    required this.depositInstallment,
    required this.paymentBalance,
    required this.paymentDate,
    required this.paymentTime,
    required this.status,
    this.failureReason,
  });

  factory PaymentInfoItem.fromJson(Map<String, dynamic> json) {
    return PaymentInfoItem(
      depositInstallment: json['depositInstallment'],
      paymentBalance: json['paymentBalance'],
      paymentDate: json['paymentDate'],
      paymentTime: json['paymentTime'],
      status: json['status'],
      failureReason: json['failureReason'],
    );
  }
}