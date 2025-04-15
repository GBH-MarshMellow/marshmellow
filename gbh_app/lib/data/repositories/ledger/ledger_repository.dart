import 'package:marshmellow/data/datasources/remote/ledger_api.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/data/models/ledger/category/transaction_category.dart';
import 'package:marshmellow/data/models/ledger/category/withdrawal_category.dart';
import 'package:marshmellow/data/models/ledger/category/deposit_category.dart';
import 'package:marshmellow/data/models/ledger/category/transfer_category.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/ledger_viewmodel.dart';
import 'package:marshmellow/data/models/ledger/payment_method.dart';

class LedgerRepository {
  final LedgerApi _ledgerApi;

  LedgerRepository(this._ledgerApi);

  // 가계부 조회
  Future<Map<String, dynamic>> getHouseholdList({
    required String startDate,
    required String endDate,
  }) async {
    return await _ledgerApi.getHouseholdList(
      startDate: startDate,
      endDate: endDate,
    );
  }

  // 카테고리 관련 메서드
  WithdrawalCategory? getWithdrawalCategoryByName(String name) {
    try {
      return WithdrawalCategory.getByName(name);
    } catch (e) {
      return null;
    }
  }

  DepositCategory? getDepositCategoryByName(String name) {
    try {
      return DepositCategory.getByName(name);
    } catch (e) {
      return null;
    }
  }

  TransferCategory? getTransferCategoryByName(String name) {
    try {
      return TransferCategory.getByName(name);
    } catch (e) {
      return null;
    }
  }

  // 거래를 날짜별로 그룹화하는 메서드
  Map<DateTime, List<Transaction>> groupTransactionsByDate(
      List<Transaction> transactions) {
    final result = <DateTime, List<Transaction>>{};

    for (var transaction in transactions) {
      final dateTime = transaction.dateTime;
      final date = DateTime(dateTime.year, dateTime.month, dateTime.day);

      if (!result.containsKey(date)) {
        result[date] = [];
      }

      result[date]!.add(transaction);
    }

    // 각 날짜 그룹 내에서 시간 역순으로 정렬 (최신순)
    result.forEach((date, items) {
      items.sort((a, b) => b.dateTime.compareTo(a.dateTime));
    });

    return result;
  }

  // 특정 날짜 범위의 거래 가져오기
  Future<List<Transaction>> getTransactions({
    required DateTime startDate,
    required DateTime endDate,
  }) async {
    // 날짜를 API에 맞는 형식으로 변환
    final formattedStartDate = DateFormat('yyyyMMdd').format(startDate);
    final formattedEndDate = DateFormat('yyyyMMdd').format(endDate);

    // API 호출 결과 가져오기
    final result = await getHouseholdList(
      startDate: formattedStartDate,
      endDate: formattedEndDate,
    );

    // 모든 거래래 반환
    return result['allTransactions'] as List<Transaction>;
  }

  // 가계부 상세 조회
  Future<Transaction> getHouseholdDetail(int householdPk) async {
    return await _ledgerApi.getHouseholdDetail(householdPk);
  }

  // 검색 기능
  Future<List<Transaction>> searchTransactions({
    required String startDate,
    required String endDate,
    required String keyword,
  }) async {
    try {
      final result = await _ledgerApi.searchHousehold(
        startDate: startDate,
        endDate: endDate,
        keyword: keyword,
      );

      return result['transactions'] as List<Transaction>;
    } catch (e) {
      throw Exception('거래 검색에 실패했습니다: $e');
    }
  }

  // 가계부 삭제
  Future<void> deleteTransaction(int householdPk) async {
    try {
      await _ledgerApi.deleteHousehold(householdPk: householdPk);
    } catch (e) {
      throw Exception('거래 내역 삭제 실패: $e');
    }
  }

  // 가계부 수정
  Future<Transaction> updateTransaction({
    required int householdPk,
    int? householdAmount,
    String? householdMemo,
    String? exceptedBudgetYn,
    int? householdDetailCategoryPk,
  }) async {
    try {
      return await _ledgerApi.updateHousehold(
        householdPk: householdPk,
        householdAmount: householdAmount,
        householdMemo: householdMemo,
        exceptedBudgetYn: exceptedBudgetYn,
        householdDetailCategoryPk: householdDetailCategoryPk,
      );
    } catch (e) {
      throw Exception('거래내역 수정 실패: $e');
    }
  }

  // 특정 분류(수입/지출/이체)에 따른 가계부 내역 조회
  Future<Map<String, dynamic>> getHouseholdByClassification({
    required String startDate,
    required String endDate,
    required String classification, // 'DEPOSIT', 'WITHDRAWAL', 'TRANSFER'
  }) async {
    try {
      final result = await _ledgerApi.getHouseholdFilter(
        startDate: startDate,
        endDate: endDate,
        classification: classification,
      );

      // 총 금액
      final total = result['total'] ?? 0;

      // 트랜잭션 목록
      final List<Transaction> transactions = [];

      // 날짜별 가계부 목록 처리
      final householdList = result['householdList'] ?? [];

      for (var dateGroup in householdList) {
        final date = dateGroup['date'] as String;
        final list = dateGroup['list'] as List;

        for (var item in list) {
          final transaction = Transaction.fromJson(item);
          transactions.add(transaction);
        }
      }

      return {
        'total': total,
        'transactions': transactions,
      };
    } catch (e) {
      throw Exception('분류별 가계부 목록을 가져오는데 실패했습니다: $e');
    }
  }

  // 결제수단 목록 조회
  Future<List<PaymentMethod>> getPaymentMethods() async {
    try {
      final response = await _ledgerApi.getPaymentMethods();
      return response.data.paymentMethodList;
    } catch (e) {
      throw Exception('결제수단 목록을 가져오는데 실패했습니다: $e');
    }
  }

  // 가계부 등록
  Future<Map<String, dynamic>> createHousehold({
    required String tradeName,
    required String tradeDate,
    required String tradeTime,
    required int householdAmount,
    String? householdMemo,
    required String paymentMethod,
    required String exceptedBudgetYn,
    required String householdClassification,
    required int householdDetailCategoryPk,
  }) async {
    try {
      return await _ledgerApi.createHousehold(
        tradeName: tradeName,
        tradeDate: tradeDate,
        tradeTime: tradeTime,
        householdAmount: householdAmount,
        householdMemo: householdMemo,
        paymentMethod: paymentMethod,
        exceptedBudgetYn: exceptedBudgetYn,
        householdClassification: householdClassification,
        householdDetailCategoryPk: householdDetailCategoryPk,
      );
    } catch (e) {
      throw Exception('가계부 등록에 실패했습니다: $e');
    }
  }
}
