import 'package:marshmellow/core/constants/icon_path.dart';

enum DepositCategoryType {
  bank,
  business,
  insurance,
  npay,
  parttime,
  realestate,
  salary,
  bonus,
  pinmoney,
  scholarship,
  etc,
}

class DepositCategory {
  final DepositCategoryType type;
  final String name;
  final String iconPath;

  const DepositCategory({
    required this.type,
    required this.name,
    required this.iconPath,
  });

  // 모든 수입 카테고리 목록
  static final List<DepositCategory> allCategories = [
    DepositCategory(
      type: DepositCategoryType.salary,
      name: '급여',
      iconPath: IconPath.incomeSalary,
    ),
    DepositCategory(
      type: DepositCategoryType.bonus,
      name: '상여금',
      iconPath: IconPath.incomeSalary,
    ),
    DepositCategory(
      type: DepositCategoryType.business,
      name: '사업수입',
      iconPath: IconPath.incomeBusiness,
    ),
    DepositCategory(
      type: DepositCategoryType.parttime,
      name: '아르바이트',
      iconPath: IconPath.incomeParttime,
    ),
    DepositCategory(
      type: DepositCategoryType.pinmoney,
      name: '용돈',
      iconPath: IconPath.incomeParttime,
    ),
    DepositCategory(
      type: DepositCategoryType.bank,
      name: '금융수입',
      iconPath: IconPath.incomeBank,
    ),
    DepositCategory(
      type: DepositCategoryType.insurance,
      name: '보험금',
      iconPath: IconPath.incomeInsurance,
    ),
    DepositCategory(
      type: DepositCategoryType.scholarship,
      name: '장학금',
      iconPath: IconPath.incomeScholarship,
    ),
    DepositCategory(
      type: DepositCategoryType.realestate,
      name: '부동산',
      iconPath: IconPath.incomeRealestate,
    ),
    DepositCategory(
      type: DepositCategoryType.npay,
      name: '더치페이',
      iconPath: IconPath.incomeNpay,
    ),
    DepositCategory(
      type: DepositCategoryType.etc,
      name: '기타수입',
      iconPath: IconPath.incomeEtc,
    ),
  ];

  // ID로 카테고리 찾기
  static DepositCategory getById(DepositCategoryType id) {
    return allCategories.firstWhere(
      (category) => category.type == id,
      orElse: () => allCategories.last, // 기본값은 '기타'
    );
  }

  // 이름으로 카테고리 찾기
  static DepositCategory getByName(String name) {
    return allCategories.firstWhere(
      (category) => category.name == name,
      orElse: () => allCategories.last, // 기본값은 '기타'
    );
  }
}
