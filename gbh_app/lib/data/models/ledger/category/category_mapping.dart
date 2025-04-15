import 'package:marshmellow/data/models/ledger/category/withdrawal_category.dart';
import 'package:marshmellow/data/models/ledger/category/deposit_category.dart';
import 'package:marshmellow/data/models/ledger/category/transfer_category.dart';
import 'package:marshmellow/data/models/ledger/category/transaction_category.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/transfer_direction_picker.dart';

/// 카테고리 타입에 따른 카테고리 PK 매핑 클래스
/// API 요청 시 필요한 householdDetailCategoryPk 값을 제공합니다.
class CategoryPkMapping {
  // 단일 인스턴스 생성 (싱글톤 패턴)
  static final CategoryPkMapping _instance = CategoryPkMapping._internal();
  
  factory CategoryPkMapping() {
    return _instance;
  }
  
  CategoryPkMapping._internal();

  /// 수입 카테고리 매핑 (DepositCategoryType -> PK)
  static const Map<DepositCategoryType, int> depositCategoryPkMap = {
    DepositCategoryType.salary: 119,      // 급여
    DepositCategoryType.bonus: 120,       // 상여금
    DepositCategoryType.business: 121,    // 사업수입
    DepositCategoryType.parttime: 122,    // 아르바이트
    DepositCategoryType.pinmoney: 123,    // 용돈
    DepositCategoryType.bank: 124,        // 금융수입
    DepositCategoryType.insurance: 125,   // 보험금
    DepositCategoryType.scholarship: 126, // 장학금
    DepositCategoryType.realestate: 127,  // 부동산
    DepositCategoryType.npay: 128,        // 더치페이
    DepositCategoryType.etc: 129,         // 기타수입
  };

  /// 출금 이체 카테고리 매핑 (TransferCategoryType -> PK)
  static const Map<TransferCategoryType, int> withdrawalTransferCategoryPkMap = {
    TransferCategoryType.internalTransfer: 130, // 내계좌이체
    TransferCategoryType.externalTransfer: 131, // 이체
    TransferCategoryType.card: 132,             // 카드대금
    TransferCategoryType.saving: 133,           // 저축
    TransferCategoryType.cash: 134,             // 현금
    TransferCategoryType.investment: 135,       // 투자
    TransferCategoryType.loan: 136,             // 대출
    TransferCategoryType.insurance: 137,        // 보험
    TransferCategoryType.etc: 138,              // 기타
  };

  /// 지출 카테고리 매핑 (WithdrawalCategoryType -> PK)
  static const Map<WithdrawalCategoryType, int> expenseCategoryPkMap = {
    WithdrawalCategoryType.alcohol: 16,         // 술/유흥
    WithdrawalCategoryType.baby: 108,           // 자녀/육아
    WithdrawalCategoryType.bank: 80,            // 금융
    WithdrawalCategoryType.car: 55,             // 자동차
    WithdrawalCategoryType.coffee: 10,          // 카페/간식
    WithdrawalCategoryType.culture: 85,         // 문화/여가
    WithdrawalCategoryType.event: 116,          // 경조/선물
    WithdrawalCategoryType.food: 1,             // 식비
    WithdrawalCategoryType.health: 65,          // 의료/건강
    WithdrawalCategoryType.house: 58,           // 주거/통신
    WithdrawalCategoryType.living: 23,          // 생활
    WithdrawalCategoryType.onlineShopping: 30,  // 온라인쇼핑
    WithdrawalCategoryType.pet: 111,            // 반려동물
    WithdrawalCategoryType.shopping: 35,        // 패션/쇼핑
    WithdrawalCategoryType.study: 101,          // 교육/학습
    WithdrawalCategoryType.transport: 47,       // 교통
    WithdrawalCategoryType.travel: 95,          // 여행/숙박
    WithdrawalCategoryType.beauty: 40,          // 뷰티/미용
    WithdrawalCategoryType.nonCategory: 118,    // 기타
  };

  /// 입금 이체 카테고리 매핑 (TransferCategoryType -> PK)
  static const Map<TransferCategoryType, int> depositTransferCategoryPkMap = {
    TransferCategoryType.internalTransfer: 139, // 내계좌이체
    TransferCategoryType.externalTransfer: 140, // 이체
    TransferCategoryType.card: 141,             // 카드대금
    TransferCategoryType.saving: 142,           // 저축
    TransferCategoryType.cash: 143,             // 현금
    TransferCategoryType.investment: 144,       // 투자
    TransferCategoryType.loan: 145,             // 대출
    TransferCategoryType.insurance: 146,        // 보험
    TransferCategoryType.etc: 147,              // 기타
  };

  /// 카테고리 객체와 이체 방향으로부터 카테고리 PK를 얻는 메서드
  static int? getPkFromCategory({
    WithdrawalCategory? expenseCategory,
    DepositCategory? incomeCategory,
    TransferCategory? transferCategory,
    TransferDirection? transferDirection,
  }) {
    if (expenseCategory != null) {
      return expenseCategoryPkMap[expenseCategory.type];
    } else if (incomeCategory != null) {
      return depositCategoryPkMap[incomeCategory.type];
    } else if (transferCategory != null && transferDirection != null) {
      if (transferDirection == TransferDirection.withdrawal) {
        return withdrawalTransferCategoryPkMap[transferCategory.type];
      } else {
        return depositTransferCategoryPkMap[transferCategory.type];
      }
    }
    return null;
  }

  /// 카테고리 이름에서 PK를 찾는 메서드
  static int? getPkFromCategoryName(String categoryName, TransactionType transactionType, {TransferDirection? transferDirection}) {
    switch (transactionType) {
      case TransactionType.withdrawal:
        final category = WithdrawalCategory.getByName(categoryName);
        return expenseCategoryPkMap[category.type];
      
      case TransactionType.deposit:
        final category = DepositCategory.getByName(categoryName);
        return depositCategoryPkMap[category.type];
      
      case TransactionType.transfer:
        final category = TransferCategory.getByName(categoryName);
        if (transferDirection == null) return null;
        
        if (transferDirection == TransferDirection.withdrawal) {
          return withdrawalTransferCategoryPkMap[category.type];
        } else {
          return depositTransferCategoryPkMap[category.type];
        }
    }
  }

  /// PK로부터 카테고리 객체를 찾는 메서드
  static dynamic getCategoryFromPk(int pk) {
    // 지출 카테고리 확인
    for (var entry in expenseCategoryPkMap.entries) {
      if (entry.value == pk) {
        return WithdrawalCategory.getById(entry.key);
      }
    }
    
    // 수입 카테고리 확인
    for (var entry in depositCategoryPkMap.entries) {
      if (entry.value == pk) {
        return DepositCategory.getById(entry.key);
      }
    }
    
    // 출금 이체 카테고리 확인
    for (var entry in withdrawalTransferCategoryPkMap.entries) {
      if (entry.value == pk) {
        return TransferCategory.getById(entry.key);
      }
    }
    
    // 입금 이체 카테고리 확인
    for (var entry in depositTransferCategoryPkMap.entries) {
      if (entry.value == pk) {
        return TransferCategory.getById(entry.key);
      }
    }
    
    return null;
  }

  /// PK가 어떤 타입의 카테고리인지 확인하는 메서드
  static TransactionType? getTransactionTypeFromPk(int pk) {
    if (expenseCategoryPkMap.values.contains(pk)) {
      return TransactionType.withdrawal;
    } else if (depositCategoryPkMap.values.contains(pk)) {
      return TransactionType.deposit;
    } else if (withdrawalTransferCategoryPkMap.values.contains(pk) || 
               depositTransferCategoryPkMap.values.contains(pk)) {
      return TransactionType.transfer;
    }
    return null;
  }
  
  /// PK가 입금 이체인지 출금 이체인지 확인하는 메서드
  static TransferDirection? getTransferDirectionFromPk(int pk) {
    if (withdrawalTransferCategoryPkMap.values.contains(pk)) {
      return TransferDirection.withdrawal;
    } else if (depositTransferCategoryPkMap.values.contains(pk)) {
      return TransferDirection.deposit;
    }
    return null;
  }
}