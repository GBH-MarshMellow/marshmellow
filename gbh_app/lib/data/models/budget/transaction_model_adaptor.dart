import 'package:marshmellow/data/models/ledger/category/transactions.dart';

// API 응답 형식을 기존 Transaction 모델로 변환하는 확장 메서드
extension HouseholdToTransaction on Map<String, dynamic> {
  Transaction toTransaction() {
    try {
      // API 응답과 Transaction 모델 간의 필드 매핑
      final householdPk = this['householdPk'] as int;
      final tradeName = this['tradeName'] as String;
      final tradeDate = this['tradeDate'] as String;
      final tradeTime = this['tradeTime'] as String;
      final householdAmount = this['householdAmount'] as int;
      
      // 선택적 필드 추출
      final householdMemo = this['householdMemo'] as String?;
      final paymentMethod = this['paymentMethod'] as String? ?? '';
      final paymentCancelYn = this['paymentCancelYn'] as String? ?? 'N';
      final exceptedBudgetYn = this['exceptedBudgetYn'] as String? ?? 'N';
      
      // 카테고리 정보 추출
      String householdCategory = '';
      String? householdDetailCategory;
      
      // 구조 1: 중첩된 구조 처리
      if (this['householdDetailCategory'] != null && 
          this['householdDetailCategory'] is Map<String, dynamic>) {
        final detailCategory = this['householdDetailCategory'] as Map<String, dynamic>;
        householdDetailCategory = detailCategory['householdDetailCategory'] as String?;
        
        if (detailCategory['householdCategory'] != null &&
            detailCategory['householdCategory'] is Map<String, dynamic>) {
          final category = detailCategory['householdCategory'] as Map<String, dynamic>;
          householdCategory = category['householdCategoryName'] as String? ?? '';
        }
      }
      
      // 가계부 분류 카테고리 (WITHDRAWAL, DEPOSIT, TRANSFER)
      final classificationStr = this['householdClassificationCategory'] as String? ?? 'WITHDRAWAL';
      final classification = _parseClassification(classificationStr);
      
      // Transaction 객체 생성
      return Transaction(
        householdPk: householdPk,
        tradeName: tradeName,
        tradeDate: tradeDate,
        tradeTime: tradeTime,
        householdAmount: householdAmount,
        householdMemo: householdMemo,
        paymentMethod: paymentMethod,
        paymentCancelYn: paymentCancelYn,
        exceptedBudgetYn: exceptedBudgetYn,
        householdCategory: householdCategory,
        householdDetailCategory: householdDetailCategory,
        classification: classification,
      );
    } catch (e) {
      print('Error in HouseholdToTransaction: $e');
      print('JSON: $this');
      rethrow;
    }
  }
  
  // 문자열을 TransactionClassification 열거형으로 변환
  static TransactionClassification _parseClassification(String value) {
    switch (value.toUpperCase()) {
      case 'DEPOSIT':
        return TransactionClassification.DEPOSIT;
      case 'WITHDRAWAL':
        return TransactionClassification.WITHDRAWAL;
      case 'TRANSFER':
        return TransactionClassification.TRANSFER;
      default:
        return TransactionClassification.WITHDRAWAL; // 기본값
    }
  }
}

// Household API 응답을 Transaction 리스트로 변환하는 함수
List<Transaction> convertHouseholdToTransactions(List<dynamic> households) {
  return households
    .map((household) => (household as Map<String, dynamic>).toTransaction())
    .toList();
}