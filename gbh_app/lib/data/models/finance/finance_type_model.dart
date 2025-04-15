// 자산 유형 분석 결과를 나타내는 모델 클래스
class FinanceTypeModel {
  final int id;           // 유형 ID (1~6)
  final String type;      // 유형 분류명
  final String nickname;  // 사용자에게 보여줄 별명
  final String condition; // 분류 조건 (문자열 설명)
  final String shortDescription; // 간략 설명
  final String longDescription;  // 상세 설명

  // 생성자
  FinanceTypeModel({
    required this.id,
    required this.type,
    required this.nickname,
    required this.condition,
    required this.shortDescription,
    required this.longDescription,
  });

  // JSON에서 객체 생성
  factory FinanceTypeModel.fromJson(Map<String, dynamic> json) {
    return FinanceTypeModel(
      id: json['id'],
      type: json['type'],
      nickname: json['nickname'],
      condition: json['condition'],
      shortDescription: json['shortDescription'],
      longDescription: json['longDescription'],
    );
  }

  // 객체를 JSON으로 변환
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'type': type,
      'nickname': nickname,
      'condition': condition,
      'shortDescription': shortDescription,
      'longDescription': longDescription,
    };
  }

  // 이미지 경로 getter (ID에 따라 다른 이미지 반환)
  String get imagePath => 'assets/images/finance/finance_analysis_card_$id.png';
}