// 입출금 내역 조회 응답 모델
class DemandDetailResponse {
  final int code;
  final String message;
  final DemandDetailData? data;

  DemandDetailResponse({
    required this.code,
    required this.message,
    this.data,
  });

  factory DemandDetailResponse.fromJson(Map<String, dynamic> json) {
    return DemandDetailResponse(
      code: json['code'],
      message: json['message'],
      data: json['data'] != null ? DemandDetailData.fromJson(json['data']) : null,
    );
  }
}

class DemandDetailData {
  //**********************************************
  //* 추가: IV 필드 추가
  //**********************************************
  final String iv;  // 암호화 IV
  final List<TransactionItem>? transactionList;

  DemandDetailData({
    required this.iv,
    this.transactionList,
  });

  factory DemandDetailData.fromJson(Map<String, dynamic> json) {
    return DemandDetailData(
      iv: json['iv'],
      transactionList: json['transactionList'] != null
          ? (json['transactionList'] as List)
              .map((item) => TransactionItem.fromJson(item))
              .toList()
          : null,
    );
  }
}

class TransactionItem {
  //**********************************************
  //* 모든 필드가 암호화되어 전달되므로 String 타입으로 정의
  //**********************************************
  final String transactionUniqueNo;
  final String transactionDate;
  final String transactionTime;
  final String transactionType;
  final String transactionTypeName;
  final String transactionAccountNo;
  final String transactionBalance;
  final String transactionAfterBalance;
  final String transactionSummary;
  final String transactionMemo;

  TransactionItem({
    required this.transactionUniqueNo,
    required this.transactionDate,
    required this.transactionTime,
    required this.transactionType,
    required this.transactionTypeName,
    required this.transactionAccountNo,
    required this.transactionBalance,
    required this.transactionAfterBalance,
    required this.transactionSummary,
    required this.transactionMemo,
  });

  factory TransactionItem.fromJson(Map<String, dynamic> json) {
    return TransactionItem(
      transactionUniqueNo: json['transactionUniqueNo'],
      transactionDate: json['transactionDate'],
      transactionTime: json['transactionTime'],
      transactionType: json['transactionType'],
      transactionTypeName: json['transactionTypeName'],
      transactionAccountNo: json['transactionAccountNo'],
      transactionBalance: json['transactionBalance'],
      transactionAfterBalance: json['transactionAfterBalance'],
      transactionSummary: json['transactionSummary'],
      transactionMemo: json['transactionMemo'],
    );
  }
}