import 'package:marshmellow/core/constants/icon_path.dart';

enum WithdrawalCategoryType {
  alcohol,
  baby,
  bank,
  car,
  coffee,
  culture,
  event,
  food,
  health,
  house,
  living,
  onlineShopping,
  pet,
  shopping,
  study,
  transport,
  travel,
  beauty,
  nonCategory,
}

class WithdrawalCategory {
  final WithdrawalCategoryType type;
  final String name;
  final String iconPath;

  const WithdrawalCategory({
    required this.type,
    required this.name,
    required this.iconPath,
  });

  // 모든 카테고리 목록
  static final List<WithdrawalCategory> allCategories = [
    WithdrawalCategory(
      type: WithdrawalCategoryType.alcohol,
      name: '술/유흥',
      iconPath: IconPath.expenseAlcohol,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.baby,
      name: '자녀/육아',
      iconPath: IconPath.expenseBaby,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.bank,
      name: '금융',
      iconPath: IconPath.expenseBank,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.car,
      name: '자동차',
      iconPath: IconPath.expenseCar,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.coffee,
      name: '카페/간식',
      iconPath: IconPath.expenseCoffee,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.culture,
      name: '문화/여가',
      iconPath: IconPath.expenseCulture,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.event,
      name: '경조/선물',
      iconPath: IconPath.expenseEvent,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.food,
      name: '식비',
      iconPath: IconPath.expenseFood,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.health,
      name: '의료/건강',
      iconPath: IconPath.expenseHealth,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.house,
      name: '주거/통신',
      iconPath: IconPath.expenseHouse,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.living,
      name: '생활',
      iconPath: IconPath.expenseLiving,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.onlineShopping,
      name: '온라인쇼핑',
      iconPath: IconPath.expenseOnlineShopping,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.pet,
      name: '반려동물',
      iconPath: IconPath.expensePet,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.shopping,
      name: '패션/쇼핑',
      iconPath: IconPath.expenseShopping,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.study,
      name: '교육/학습',
      iconPath: IconPath.expenseStudy,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.transport,
      name: '교통',
      iconPath: IconPath.expenseTransport,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.travel,
      name: '여행/숙박',
      iconPath: IconPath.expenseTravel,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.beauty,
      name: '뷰티/미용',
      iconPath: IconPath.expenseBeauty,
    ),
    WithdrawalCategory(
      type: WithdrawalCategoryType.nonCategory,
      name: '기타',
      iconPath: IconPath.nonCategory,
    ),
  ];

  // ID로 카테고리 찾기
  static WithdrawalCategory getById(WithdrawalCategoryType id) {
    return allCategories.firstWhere(
      (category) => category.type == id,
      orElse: () => allCategories.last, // 기본값은 '기타'
    );
  }

  // 이름으로 카테고리 찾기
  static WithdrawalCategory getByName(String name) {
    return allCategories.firstWhere(
      (category) => category.name == name,
      orElse: () => allCategories.last, // 기본값은 '기타'
    );
  }
}
