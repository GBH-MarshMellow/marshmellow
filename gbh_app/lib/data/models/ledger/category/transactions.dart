import 'package:marshmellow/data/models/ledger/category/transaction_category.dart';

enum TransactionClassification {
  DEPOSIT, // 수입
  WITHDRAWAL, // 지출
  TRANSFER // 이체
}

class Transaction {
  final int householdPk; // 가계부 고유번호
  final String tradeName; // 거래처 명
  final String tradeDate; // 거래일 (yyyyMMdd 형식)
  final String tradeTime; // 거래시간 (hhMM 형식)
  final int householdAmount; // 사용금액
  final String? householdMemo; // 메모 (API에서는 없지만 앱에서 필요할 수 있음)
  final String paymentMethod; // 결제수단
  final String paymentCancelYn; // 거래 취소 여부 (Y 또는 N)
  final String exceptedBudgetYn; // 예산 제외 여부 (Y 또는 N)
  final String householdCategory; // 가계부 메인 카테고리
  final String?
      householdDetailCategory; // 가계부 상세 카테고리 (API에서는 없지만 앱에서 필요할 수 있음)
  final TransactionClassification
      classification; // 가계부 분류 (DEPOSIT, WITHDRAWAL, TRANSFER)

  Transaction({
    required this.householdPk,
    required this.tradeName,
    required this.tradeDate,
    required this.tradeTime,
    required this.householdAmount,
    this.householdMemo,
    required this.paymentMethod,
    required this.paymentCancelYn,
    required this.householdCategory,
    this.householdDetailCategory,
    required this.classification,
    required this.exceptedBudgetYn,
  });

  // API 응답에서 Transaction 객체 생성 - 명세서에 맞게 수정
  factory Transaction.fromJson(Map<String, dynamic> json) {
    try {
      // API에서 반환되는 필드 중 일부를 안전하게 추출
      final householdPk = json['householdPk'] as int;
      final tradeName = json['tradeName'] as String;
      final tradeDate = json['tradeDate'] as String;
      final tradeTime = json['tradeTime'] as String;
      final householdAmount = json['householdAmount'] as int;

      // 선택적 필드는 안전하게 추출
      final householdMemo = json['householdMemo'] as String?;
      final paymentMethod = json['paymentMethod'] as String;
      final paymentCancelYn = json['paymentCancelYn'] as String;
      final exceptedBudgetYn = json['exceptedBudgetYn'] as String? ?? 'N';

      // 카테고리 정보 추출 - 두 가지 형태 모두 처리
      String householdCategory = '';
      String? householdDetailCategory;

      // 구조 1: search API 형태 (중첩 객체)
      if (json['householdDetailCategory'] != null &&
          json['householdDetailCategory'] is Map) {
        final detailCategory = json['householdDetailCategory'] as Map;
        householdDetailCategory =
            detailCategory['householdDetailCategory'] as String?;

        if (detailCategory['householdCategory'] != null &&
            detailCategory['householdCategory'] is Map) {
          final category = detailCategory['householdCategory'] as Map;
          householdCategory =
              category['householdCategoryName'] as String? ?? '';
        }
      }
      // 구조 2: list API 형태 (플랫 구조)
      else if (json['householdCategory'] != null) {
        householdCategory = json['householdCategory'] as String;
        householdDetailCategory = json['householdDetailCategory'] as String?;
      }

      // 분류 카테고리 추출
      final classificationStr =
          json['householdClassificationCategory'] as String;
      final classification = _parseClassification(classificationStr);

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
      print('Error in Transaction.fromJson: $e');
      print('JSON: $json');
      rethrow; // 디버깅을 위해 예외 다시 발생
    }
  }

  // 문자열을 TransactionClassification 열거형으로 변환
  static TransactionClassification _parseClassification(String value) {
    switch (value) {
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

  // DateTime 형식으로 거래일자 반환 (필요시 사용)
  DateTime get dateTime {
    final year = int.parse(tradeDate.substring(0, 4));
    final month = int.parse(tradeDate.substring(4, 6));
    final day = int.parse(tradeDate.substring(6, 8));
    final hour = int.parse(tradeTime.substring(0, 2));
    final minute = int.parse(tradeTime.substring(2, 4));

    return DateTime(year, month, day, hour, minute);
  }

  // JSON으로 변환
  Map<String, dynamic> toJson() {
    return {
      'householdPk': householdPk,
      'tradeName': tradeName,
      'tradeDate': tradeDate,
      'tradeTime': tradeTime,
      'householdAmount': householdAmount,
      'householdMemo': householdMemo,
      'paymentMethod': paymentMethod,
      'paymentCancelYn': paymentCancelYn,
      'exceptedBudgetYn': exceptedBudgetYn,
      'householdCategory': householdCategory,
      'householdDetailCategory': householdDetailCategory,
      'householdClassificationCategory':
          _classificationToString(classification),
    };
  }

  // TransactionClassification 열거형을 문자열로 변환
  static String _classificationToString(
      TransactionClassification classification) {
    switch (classification) {
      case TransactionClassification.DEPOSIT:
        return 'DEPOSIT';
      case TransactionClassification.WITHDRAWAL:
        return 'WITHDRAWAL';
      case TransactionClassification.TRANSFER:
        return 'TRANSFER';
    }
  }

  // TransactionListItem 위젯과 호환되도록 추가하는 getter 메서드들
  String get title => tradeName;
  double get amount => householdAmount.toDouble();
  String? get accountName => null; // API에 없는 정보는 null 반환

// 트랜잭션 타입 반환 (TransactionCategory 클래스와 호환되도록)
  TransactionType get type {
    switch (classification) {
      case TransactionClassification.DEPOSIT:
        return TransactionType.deposit;
      case TransactionClassification.WITHDRAWAL:
        return TransactionType.withdrawal;
      case TransactionClassification.TRANSFER:
        return TransactionType.transfer;
    }
  }

// 카테고리 ID 반환 (TransactionListItem에서 필요)
  String get categoryId => householdCategory;

// 트랜잭션 ID (TransactionListItem에서 필요)
  int get id => householdPk;
}
