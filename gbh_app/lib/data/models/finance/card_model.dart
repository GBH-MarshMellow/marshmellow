// 카드 관련 모델
class CardData {
  //**********************************************
  //* 변경: totalAmount를 String 타입으로 변경 (암호화된 값이므로)
  //**********************************************
  final String totalAmount;  // int에서 String으로 변경
  final List<CardItem> cardList;

  CardData({
    required this.totalAmount,
    required this.cardList,
  });

  factory CardData.fromJson(Map<String, dynamic> json) {
    return CardData(
      //**********************************************
      //* 변경: 원래 정수형이었던 totalAmount가 이제는 암호화된 문자열
      //**********************************************
      totalAmount: json['totalAmount'],
      cardList: (json['cardList'] as List)
          .map((item) => CardItem.fromJson(item))
          .toList(),
    );
  }
}

class CardItem {
  final String cardNo;
  final String cvc;
  final String cardIssuerCode;
  final String cardIssuerName;
  final String cardName;
  //**********************************************
  //* 변경: cardBalance를 String 타입으로 변경 (암호화된 값이므로)
  //**********************************************
  final String? cardBalance;  // int에서 String으로 변경

  CardItem({
    required this.cardNo,
    required this.cvc,
    required this.cardIssuerCode,
    required this.cardIssuerName,
    required this.cardName,
    this.cardBalance,
  });

  factory CardItem.fromJson(Map<String, dynamic> json) {
    return CardItem(
      cardNo: json['cardNo'],
      cvc: json['cvc'],
      cardIssuerCode: json['cardIssuerCode'],
      cardIssuerName: json['cardIssuerName'],
      cardName: json['cardName'],
      //**********************************************
      //* 변경: 원래 정수형이었던 cardBalance가 이제는 암호화된 문자열
      //**********************************************
      cardBalance: json['cardBalance'],
    );
  }
}