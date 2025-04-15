// 송금 요청 모델
class TransferRequest {
  // IV 필드를 추가
  final String? iv;  // 암호화에 사용된 IV (인터셉터에서 자동 생성)
  final int withdrawalAccountId;
  final String depositAccountNo;
  final String transactionSummary;
  // 타입을 int에서 String으로 변경
  final String transactionBalance;  // 암호화 형태로 전송되므로 String 타입으로 변경

  TransferRequest({
    this.iv,  // 옵션으로 추가 (인터셉터에서 자동으로 처리하기 때문)
    required this.withdrawalAccountId,
    required this.depositAccountNo,
    required this.transactionSummary,
    required this.transactionBalance,
  });

  Map<String, dynamic> toJson() {
    return {
      // iv는 인터셉터에서 자동으로 처리하므로 추가하지 않음
      'withdrawalAccountId': withdrawalAccountId,
      'depositAccountNo': depositAccountNo,
      'transactionSummary': transactionSummary,
      'transactionBalance': transactionBalance,
    };
  }
}

// 송금 응답 모델
// 송금 응답 모델 수정
class TransferResponse {
  final int code;
  final String message;
  final dynamic data; // Map이나 List 둘 다 가능하도록 dynamic으로 변경

  TransferResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory TransferResponse.fromJson(Map<String, dynamic> json) {
    return TransferResponse(
      code: json['code'],
      message: json['message'],
      data: json['data'], // 그대로 사용
    );
  }
  
  // data가 Map인 경우 TransferData로 변환
  TransferData? get transferData {
    if (data is Map<String, dynamic>) {
      return TransferData.fromJson(data);
    }
    return null;
  }
}
// 송금 응답 데이터
class TransferData {
  final String message;

  TransferData({
    required this.message,
  });

  factory TransferData.fromJson(Map<String, dynamic> json) {
    return TransferData(
      message: json['message'],
    );
  }
}

// 은행 모델
class Bank {
  final String code;
  final String name;
  final String iconPath;

  Bank({
    required this.code,
    required this.name,
    required this.iconPath,
  });
}