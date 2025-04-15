// lib/data/models/ledger/payment_method_response.dart

import 'package:marshmellow/data/models/ledger/payment_method.dart';

class PaymentMethodResponse {
  final int code;
  final String message;
  final PaymentMethodData data;

  PaymentMethodResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory PaymentMethodResponse.fromJson(Map<String, dynamic> json) {
    return PaymentMethodResponse(
      code: json['code'] as int,
      message: json['message'] as String,
      data: PaymentMethodData.fromJson(json['data']),
    );
  }
}

class PaymentMethodData {
  final List<PaymentMethod> paymentMethodList;

  PaymentMethodData({required this.paymentMethodList});

  factory PaymentMethodData.fromJson(Map<String, dynamic> json) {
    final list = (json['paymentMethodList'] as List)
        .map((item) => PaymentMethod.fromJson(item))
        .toList();
    return PaymentMethodData(paymentMethodList: list);
  }
}