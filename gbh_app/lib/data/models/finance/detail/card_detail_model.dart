// lib/data/models/finance/card_detail_model.dart

class CardDetailResponse {
  final int code;
  final String message;
  final CardDetailData data;

  CardDetailResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory CardDetailResponse.fromJson(Map<String, dynamic> json) {
    // data가 배열인 경우 기본값 사용
    if (json['data'] is List) {
      return CardDetailResponse(
        code: json['code'],
        message: json['message'],
        data: CardDetailData(
          iv: '',
          estimatedBalance: '0',
          transactionList: []
        ),
      );
    }
    
    return CardDetailResponse(
      code: json['code'],
      message: json['message'],
      data: CardDetailData.fromJson(json['data']),
    );
  }
}

class CardDetailData {
  // IV 필드 추가
  final String iv;
  // estimatedBalance를 String으로 변경 (암호화된 값)
  final String estimatedBalance;
  final List<CardTransactionItem> transactionList;

  CardDetailData({
    required this.iv,
    required this.estimatedBalance,
    required this.transactionList,
  });

  factory CardDetailData.fromJson(Map<String, dynamic> json) {
    return CardDetailData(
      iv: json['iv'],
      estimatedBalance: json['estimatedBalance'],
      transactionList: (json['transactionList'] as List?)
          ?.map((item) => CardTransactionItem.fromJson(item))
          .toList() ?? [], // null이면 빈 리스트 반환
    );
  }
}

class CardTransactionItem {
  final String? transactionUniqueNo;
  final String? merchantId;
  // billStatementsStatus 필드는 응답에 없음 - 제거하거나 nullable로 변경
  final String? billStatementsStatus;
  final String? billStatementsYn;
  final String? transactionBalance;
  final String? transactionDate;
  final String? transactionTime;
  final String? categoryName;
  final String? categoryId;
  final String? cardStatus;
  final String? merchantName;

  CardTransactionItem({
    this.transactionUniqueNo,
    this.merchantId,
    this.billStatementsStatus,
    this.billStatementsYn,
    this.transactionBalance,
    this.transactionDate,
    this.transactionTime,
    this.categoryName,
    this.categoryId,
    this.cardStatus,
    this.merchantName,
  });

  factory CardTransactionItem.fromJson(Map<String, dynamic> json) {
    return CardTransactionItem(
      transactionUniqueNo: json['transactionUniqueNo'],
      merchantId: json['merchantId'],
      // billStatementsStatus 필드가 응답에 없으면 null로 설정
      billStatementsStatus: json['billStatementsStatus'],
      billStatementsYn: json['billStatementsYn'],
      transactionBalance: json['transactionBalance'],
      transactionDate: json['transactionDate'],
      transactionTime: json['transactionTime'],
      categoryName: json['categoryName'],
      categoryId: json['categoryId'],
      cardStatus: json['cardStatus'],
      merchantName: json['merchantName'],
    );
  }
}