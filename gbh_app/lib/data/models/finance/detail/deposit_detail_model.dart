// lib/data/models/finance/detail/deposit_detail_model.dart
class DepositDetailResponse {
  final int code;
  final String message;
  final DepositDetailData data;

  DepositDetailResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory DepositDetailResponse.fromJson(Map<String, dynamic> json) {
    return DepositDetailResponse(
      code: json['code'],
      message: json['message'],
      data: DepositDetailData.fromJson(json['data']),
    );
  }
}

class DepositDetailData {
  final PaymentItem payment;

  DepositDetailData({
    required this.payment,
  });

  factory DepositDetailData.fromJson(Map<String, dynamic> json) {
    return DepositDetailData(
      payment: PaymentItem.fromJson(json['payment']),
    );
  }
}

class PaymentItem {
  final String paymentUniqueNo;
  final String paymentDate;
  final String paymentTime;
  final String paymentBalance;

  PaymentItem({
    required this.paymentUniqueNo,
    required this.paymentDate,
    required this.paymentTime,
    required this.paymentBalance,
  });

  factory PaymentItem.fromJson(Map<String, dynamic> json) {
    return PaymentItem(
      paymentUniqueNo: json['paymentUniqueNo'],
      paymentDate: json['paymentDate'],
      paymentTime: json['paymentTime'],
      paymentBalance: json['paymentBalance'],
    );
  }
}