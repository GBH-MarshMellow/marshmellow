import 'package:marshmellow/core/constants/icon_path.dart';

enum TransferCategoryType {
  internalTransfer,
  externalTransfer,
  card,
  saving,
  cash,
  investment,
  loan,
  insurance,
  etc,
}

class TransferCategory {
  final TransferCategoryType type;
  final String name;
  final String iconPath;

  const TransferCategory({
    required this.type,
    required this.name,
    required this.iconPath,
  });
  // 모든 지출 카테고리 목록
  static final List<TransferCategory> allCategories = [
    TransferCategory(
      type: TransferCategoryType.internalTransfer,
      name: '내계좌이체',
      iconPath: IconPath.transferEtc,
    ),
    TransferCategory(
      type: TransferCategoryType.externalTransfer,
      name: '이체',
      iconPath: IconPath.transferEtc,
    ),
    TransferCategory(
      type: TransferCategoryType.card,
      name: '카드대금',
      iconPath: IconPath.transferEtc,
    ),
    TransferCategory(
      type: TransferCategoryType.saving,
      name: '저축',
      iconPath: IconPath.transferEtc,
    ),
    TransferCategory(
      type: TransferCategoryType.cash,
      name: '현금',
      iconPath: IconPath.transferEtc,
    ),
    TransferCategory(
      type: TransferCategoryType.investment,
      name: '투자',
      iconPath: IconPath.transferEtc,
    ),
    TransferCategory(
      type: TransferCategoryType.loan,
      name: '대출',
      iconPath: IconPath.transferEtc,
    ),
    TransferCategory(
      type: TransferCategoryType.insurance,
      name: '보험',
      iconPath: IconPath.transferEtc,
    ),
    TransferCategory(
      type: TransferCategoryType.etc,
      name: '기타',
      iconPath: IconPath.transferEtc,
    ),
  ];

  // ID로 카테고리 찾기
  static TransferCategory getById(TransferCategoryType id) {
    return allCategories.firstWhere(
      (category) => category.type == id,
      orElse: () => allCategories.last, // 기본값은 '기타'
    );
  }

  // 이름으로 카테고리 찾기
  static TransferCategory getByName(String name) {
    return allCategories.firstWhere(
      (category) => category.name == name,
      orElse: () => allCategories.last, // 기본값은 '기타'
    );
  }
}
