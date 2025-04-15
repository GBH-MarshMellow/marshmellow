import 'package:marshmellow/data/models/finance/finance_type_model.dart';

// 금융 유형 분석에 사용할 유형 데이터 정의
class FinanceTypeConstants {
  // 싱글톤 패턴 구현
  static final FinanceTypeConstants _instance = FinanceTypeConstants._internal();
  factory FinanceTypeConstants() => _instance;
  FinanceTypeConstants._internal();

  // 모든 유형 데이터 목록 (JSON 형태로 정의)
  static final List<Map<String, dynamic>> _financeTypesJson = [
{
      "id": 1,
      "type": "주택청약 꾸준히넣음",
      "nickname": "내 집 마련 꿈나무",
      "condition": "예금,적금,대출 중 \"주택청약\"을 포함하는 이름이 존재할 때",
      "shortDescription": "내 집 직접 건설 중",
      "longDescription": "주택청약 통장에 꾸준히 돈을 쌓아가며 벽돌 한 장씩 내 집을 직접 건설 중이시네요.\n청약 가점을 높이는 것은 마치 더 튼튼한 자재를 사용하는 것과 같아요.\n조금 더 인내심을 갖고 건설을 이어가면 언젠가 당신만의 집이 완성될 거예요. 벽돌 쌓기를 계속하세요!"
    },
    {
      "id": 2,
      "type": "소액대출",
      "nickname": "소액대출 중독자",
      "condition": "500만원 이하의 대출 보유 중일 때",
      "shortDescription": "소액으로 빚만 쌓이는 중",
      "longDescription": "\"그깟 얼마 안 되는 돈인데\"라고 생각하며 충동적으로 대출을 받고 있어요.\n작은 금액이라도 계속 쌓이면 결국 큰 부담이 됩니다. 소비습관을 점검하고 계획적인 소비를 실천해보세요.\n오늘의 5만원 대출이 내일의 500만원 빚더미가 될 수 있어요."
    },
    {
      "id": 3,
      "type": "소액 적금 많음",
      "nickname": "적금 소작농",
      "condition": "1,000,000원 이하의 소액 적금 3개보유 중일 때",
      "shortDescription": "여기저기 조금씩 모으는 중",
      "longDescription": "소액 적금을 여러개 운영중이시네요.\n목적별로 구분해서 관리하는 방식이 계획적인 소비에 도움이 될 수 있지만, 너무 많은 적금은 관리가 어려울 수 있어요.\n관리의 효율성을 위해 비슷한 목적의 적금을 통합하고, 금리가 높은 상품 위주로 정리하는 것이 이자수익을 높이는 데 도움이 될 거예요.\n또한 자동이체 설정으로 관리 부담을 줄여보세요"
    },
    {
      "id": 4,
      "type": "카드 개수가 많음",
      "nickname": "거의 유희왕",
      "condition": "카드 개수 4개 이상일 때",
      "shortDescription": "쓸대없이 카드만 많음",
      "longDescription": "Marshmallow사용자 대비 2.6장의 카드를 더 보유중이에요. 사용하지 않는 카드는 정리하는게 어떨가요?\n혜택별로 카드를 구분해서 사용하는 습관은 좋지만, 연회비와 관리 측면에서 사용하지 않는 카드는 정리하는게 어떨가요?\n주요 소비처에 맞는 2-3장의 카드만 남기는 것이 효율적일 수 있어요."
    },
    {
      "id": 5,
      "type": "적금 비율 높음",
      "nickname": "적금 라푼젤",
      "condition": "적금의 개수가 예금보다 많을 때",
      "shortDescription": "미래를 위해 쌓아두는 중",
      "longDescription": "돈을 적금에 너무 많이 묶어두고 계시네요.\n돈을 탑 안에 가두고 있어요!\n중도해지 시 이자 손해가 크기 때문에 꺼내쓰기 어려운 상황이지만, 갑작스러운 지출에 대비해 일부는 자유롭게 사용할 수 있는 예금으로 전환하는 것이 필요할 수도 있어요. 돈을 탑에서 풀어주세요!"
    },
    {
      "id": 6,
      "type": "총 예적금 금액 상위 10%",
      "nickname": "재벌집 막내아들",
      "condition": "총 예적금 1억 이상일 때",
      "shortDescription": "자산가의 풍모를 갖춤",
      "longDescription": "축하드려요! 당신의 예적금 자산은 최상위에 해당해요.\n이렇게 모인 자금을 더 효율적으로 운용하기 위해 다양한 투자 포트폴리오도 고려해보세요.\n예적금 외에도 ETF, 채권, 부동산 등 자산을 분산 투자하면 인플레이션에 대비하고 더 높은 수익률을 기대할 수 있어요.\n안정적인 노후를 위한 준비도 함께 생각해보세요."
    }
  ];

  // JSON을 FinanceTypeModel 객체 리스트로 변환
  static List<FinanceTypeModel> getAllTypes() {
    return _financeTypesJson
        .map((json) => FinanceTypeModel.fromJson(json))
        .toList();
  }

  // ID로 특정 유형 찾기
  static FinanceTypeModel getTypeById(int id) {
    return getAllTypes().firstWhere(
      (type) => type.id == id,
      orElse: () => throw Exception('Finance type with id $id not found'),
    );
  }
}