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

class PaymentMethod {
  final String bankCode;
  final String bankName;
  final String paymentType;
  final String paymentMethod;

  PaymentMethod({
    required this.bankCode,
    required this.bankName,
    required this.paymentType,
    required this.paymentMethod,
  });

  factory PaymentMethod.fromJson(Map<String, dynamic> json) {
    return PaymentMethod(
      bankCode: json['bankCode'] as String,
      bankName: json['bankName'] as String,
      paymentType: json['paymentType'] as String,
      paymentMethod: json['paymentMethod'] as String,
    );
  }
}
