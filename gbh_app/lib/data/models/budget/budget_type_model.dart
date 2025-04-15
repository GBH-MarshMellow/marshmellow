import 'package:flutter/material.dart';

/*
  예산 유형별 데이터 저장 모델
*/
class BudgetTypeData {
  final double fixedExpense;
  final double foodExpense;
  final double transportationExpense;
  final double marketExpense;
  final double financialExpense;
  final double leisureExpense;
  final double coffeeExpense;
  final double shoppingExpense;
  final double emergencyExpense;

  BudgetTypeData({
    required this.fixedExpense,
    required this.foodExpense,
    required this.transportationExpense,
    required this.marketExpense,
    required this.financialExpense,
    required this.leisureExpense,
    required this.coffeeExpense,
    required this.shoppingExpense,
    required this.emergencyExpense,
  });

  factory BudgetTypeData.fromJson(Map<String, dynamic> json) {
    return BudgetTypeData(
      fixedExpense: json['고정지출'] as double? ?? 0.0,
      foodExpense: json['식비/외식'] as double? ?? 0.0,
      transportationExpense: json['교통/자동차'] as double? ?? 0.0,
      marketExpense: json['편의점/마트'] as double? ?? 0.0,
      financialExpense: json['금융'] as double? ?? 0.0,
      leisureExpense: json['여가비'] as double? ?? 0.0,
      coffeeExpense: json['커피/디저트'] as double? ?? 0.0,
      shoppingExpense: json['쇼핑'] as double? ?? 0.0,
      emergencyExpense: json['비상금'] as double? ?? 0.0,
    );
  }

  Map<String, double> toMap() {
    return {
      '고정지출': fixedExpense,
      '식비/외식': foodExpense,
      '교통/자동차': transportationExpense,
      '편의점/마트': marketExpense,
      '금융': financialExpense,
      '여가비': leisureExpense,
      '커피/디저트': coffeeExpense,
      '쇼핑': shoppingExpense,
      '비상금': emergencyExpense,
    };
  }
}

/*
  예산 유형별 분석 응답 모델
*/
class BudgetTypeAnalysisResponse {
  final Map<String, BudgetTypeData> myData;
  final Map<String, BudgetTypeData> allData;

  BudgetTypeAnalysisResponse({
    required this.myData,
    required this.allData,
  });

  factory BudgetTypeAnalysisResponse.fromJson(Map<String, dynamic> json) {
    // myData 파싱
    final myDataJson = json['my_data'] as Map<String, dynamic>;
    final myDataMap = <String, BudgetTypeData>{};
    
    myDataJson.forEach((key, value) {
      print('📊 my_data 키: $key, 값: $value');
      myDataMap[key] = BudgetTypeData.fromJson(value as Map<String, dynamic>);
    });

    // allData 파싱
    final allDataJson = json['all_data'] as Map<String, dynamic>;
    print('📊 all_data: $allDataJson');
    final allDataMap = <String, BudgetTypeData>{};
    
    allDataJson.forEach((key, value) {
      print('📊 all_data 키: $key');
      allDataMap[key] = BudgetTypeData.fromJson(value as Map<String, dynamic>);
    });

    return BudgetTypeAnalysisResponse(
      myData: myDataMap,
      allData: allDataMap,
    );
  }
}

/*
  예산 유형별 분석 요청 모델
*/
class BudgetTypeAnalysisRequest {
  final int salary;
  final double fixedExpense;
  final double foodExpense;
  final double transportationExpense;
  final double marketExpense;
  final double financialExpense;
  final double leisureExpense;
  final double coffeeExpense;
  final double shoppingExpense;
  final double emergencyExpense;

  BudgetTypeAnalysisRequest({
    required this.salary,
    required this.fixedExpense,
    required this.foodExpense,
    required this.transportationExpense,
    required this.marketExpense,
    required this.financialExpense,
    required this.leisureExpense,
    required this.coffeeExpense,
    required this.shoppingExpense,
    required this.emergencyExpense,
  });

  Map<String, dynamic> toJson() {
    return {
      'salary': salary,
      'fixed_expense': fixedExpense,
      'food_expense': foodExpense,
      'transportation_expense': transportationExpense,
      'market_expense': marketExpense,
      'financial_expense': financialExpense,
      'leisure_expense': leisureExpense,
      'coffee_expense': coffeeExpense,
      'shopping_expense': shoppingExpense,
      'emergency_expense': emergencyExpense,
    };
  }
}

/*
  예산 유형별 정보 모델
*/
class BudgetTypeInfo {
  final String type;
  final String typeName;
  final String selectionName;
  final String description;
  final String assetPath;
  final Color color;

  BudgetTypeInfo({
    required this.type,
    required this.typeName,
    required this.selectionName,
    required this.description,
    required this.assetPath,
    required this.color,
  });

  // 타입에 따른 정보 반환
  static BudgetTypeInfo getTypeInfo(String type) {
    print('🔍 유형 정보 요청: $type');
    switch (type) {
      case '식비/외식':
        return BudgetTypeInfo(
          type: type,
          typeName: '아기 돼지 삼형제 형',
          selectionName: '미식가 형',
          description: '🍚오늘도 밥심으로 하루를 버티는 당신, 맛집 리스트는 이미 완성 단계!',
          assetPath: 'assets/images/budget/type_food.png',
          color: const Color(0xFFFFC107),
        );
      case '교통/자동차':
        return BudgetTypeInfo(
          type: type,
          typeName: '바쁘다 바빠 현대사회 형',
          selectionName: '시간은 돈이다 형',
          description: '🚗당신의 출퇴근길은 F1 급! 붕붕이 없인 못 살아~',
          assetPath: 'assets/images/budget/type_transportation.png',
          color: const Color(0xFF2196F3),
        );
      case '편의점/마트':
        return BudgetTypeInfo(
          type: type,
          typeName: '만능소비 형',
          selectionName: '생활의 지혜 형',
          description: '🎈편의점 앞에만 가면 텐션업! 소소하지만 강력한 소비자~',
          assetPath: 'assets/images/budget/type_market.png',
          color: const Color(0xFFE91E63),
        );
      case '금융':
        return BudgetTypeInfo(
          type: type,
          typeName: '월급스쳐간 형',
          selectionName: '재테크 실험가 형',
          description: '💳월급은 통장을 스치고, 카드 명세서는 소설처럼 흥미진진!',
          assetPath: 'assets/images/budget/type_finance.png',
          color: const Color(0xFF0D47A1),
        );
      case '여가비':
        return BudgetTypeInfo(
          type: type,
          typeName: '워라밸 챙기는 유형',
          selectionName: '문화 감성 충전 형',
          description: '🍀삶은 즐기라고 있는 것! 취미와 여가로 완성되는 나만의 일상',
          assetPath: 'assets/images/budget/type_leisure.png',
          color: const Color(0xFF4CAF50),
        );
      case '커피/디저트':
        return BudgetTypeInfo(
          type: type,
          typeName: '디저트배는 따로있어 형',
          selectionName: '소확행 실천 형',
          description: '☕당이 떨어지면 나도 떨어진다! 카페는 제2의 집!',
          assetPath: 'assets/images/budget/type_coffee.png',
          color: const Color(0xFF795548),
        );
      case '쇼핑':
        return BudgetTypeInfo(
          type: type,
          typeName: '지름신 형',
          selectionName: '셀프 보상 전문가 형',
          description: '🛒배송 중독자 등판! \'결제완료\'는 당신의 힐링 키워드~',
          assetPath: 'assets/images/budget/type_shopping.png',
          color: const Color(0xFFFF4081),
        );
      case '비상금':
        return BudgetTypeInfo(
          type: type,
          typeName: '지출은 갑자기 형',
          selectionName: '의리충만 배려왕 형',
          description: '💥달마다 찾아오는 예상 못한 펀치! 하지만 의리 있게 처리 완료~',
          assetPath: 'assets/images/budget/type_emergency.png',
          color: const Color(0xFF607D8B),
        );
      case '평균':
        return BudgetTypeInfo(
          type: type,
          typeName: '밸런스 게임 승자 형',
          selectionName: '현명한 소비자 형',
          description: '✨모든 영역에서 균형 잡힌 소비를 추구하는 완벽주의자',
          assetPath: 'assets/images/budget/type_average.png',
          color: const Color(0xFF9E9E9E),
        );
      default:
        print('🔍 기본 유형(평균) 정보 반환, 요청된 유형: $type');
        return BudgetTypeInfo(
          type: '평균',
          typeName: '밸런스 게임 승자 형',
          selectionName: '현명한 소비자 형',
          description: '✨모든 영역에서 균형 잡힌 소비를 추구하는 완벽주의자',
          assetPath: 'assets/images/budget/type_average.png',
          color: const Color(0xFF9E9E9E),
        );
    }
  }

  // 모든 타입 정보 반환 (선택 화면용)
  static List<BudgetTypeInfo> getAllTypeInfos() {
    final types = [
      '식비/외식',
      '교통/자동차',
      '편의점/마트',
      '금융',
      '여가비',
      '커피/디저트',
      '쇼핑',
      '비상금',
      '평균',
    ];

    return types.map((type) => getTypeInfo(type)).toList();
  }
}