import 'package:marshmellow/data/models/ledger/category/withdrawal_category.dart';
import 'package:marshmellow/data/models/ledger/category/deposit_category.dart';
import 'package:marshmellow/data/models/ledger/category/transfer_category.dart';

enum TransactionType {
  withdrawal,
  deposit,
  transfer,
}

class TransactionCategory {
  final TransactionType type;
  final dynamic categoryId; // ExpenseCategoryType 또는 IncomeCategoryType 또는 TransferCategoryType
  final String name;
  final String iconPath;

  TransactionCategory({
    required this.type,
    required this.categoryId,
    required this.name,
    required this.iconPath,
  });

  // 지출 카테고리로부터 변환
  factory TransactionCategory.fromWithdrawal(WithdrawalCategory category) {
    return TransactionCategory(
      type: TransactionType.withdrawal,
      categoryId: category.type,
      name: category.name,
      iconPath: category.iconPath,
    );
  }

  // 수입 카테고리로부터 변환
  factory TransactionCategory.fromDeposit(DepositCategory category) {
    return TransactionCategory(
      type: TransactionType.deposit,
      categoryId: category.type,
      name: category.name,
      iconPath: category.iconPath,
    );
  }

  // 이체 카테고리로부터 변환
  factory TransactionCategory.fromTransfer(TransferCategory category) {
    return TransactionCategory(
      type: TransactionType.transfer,
      categoryId: category.type,
      name: category.name,
      iconPath: category.iconPath,
    );
  }

  // 모든 지출 카테고리 가져오기
  static List<TransactionCategory> getAllWithdrawalCategories() {
    return WithdrawalCategory.allCategories
        .map((e) => TransactionCategory.fromWithdrawal(e))
        .toList();
  }

  // 모든 수입 카테고리 가져오기
  static List<TransactionCategory> getAllDepositCategories() {
    return DepositCategory.allCategories
        .map((e) => TransactionCategory.fromDeposit(e))
        .toList();
  }

  // 모든 이체 카테고리 가져오기
  static List<TransactionCategory> getAllTransferCategories() {
    return TransferCategory.allCategories
        .map((e) => TransactionCategory.fromTransfer(e))
        .toList();
  }
}
