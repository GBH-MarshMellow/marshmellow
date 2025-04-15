import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/repositories/ledger/ledger_repository.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:syncfusion_flutter_datepicker/datepicker.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';

// 가계부 상태 정의
class LedgerState {
  final bool isLoading;
  final List<Transaction> transactions;
  final Map<String, List<Transaction>> groupedTransactions; // 날짜별 그룹화된 거래
  final int totalIncome;
  final int totalExpenditure;
  final String? errorMessage;

  LedgerState({
    this.isLoading = false,
    this.transactions = const [],
    this.groupedTransactions = const {},
    this.totalIncome = 0,
    this.totalExpenditure = 0,
    this.errorMessage,
  });

  LedgerState copyWith({
    bool? isLoading,
    List<Transaction>? transactions,
    Map<String, List<Transaction>>? groupedTransactions,
    int? totalIncome,
    int? totalExpenditure,
    String? errorMessage,
  }) {
    return LedgerState(
      isLoading: isLoading ?? this.isLoading,
      transactions: transactions ?? this.transactions,
      groupedTransactions: groupedTransactions ?? this.groupedTransactions,
      totalIncome: totalIncome ?? this.totalIncome,
      totalExpenditure: totalExpenditure ?? this.totalExpenditure,
      errorMessage: errorMessage,
    );
  }
}

// 가계부 뷰모델
class LedgerViewModel extends StateNotifier<LedgerState> {
  final LedgerRepository _repository;

  LedgerViewModel(this._repository) : super(LedgerState());

  // 날짜 형식 변환 헬퍼 메서드
  String _formatDate(DateTime date) {
    return DateFormat('yyyyMMdd').format(date);
  }

  // 가계부 목록 로드
  Future<void> loadHouseholdData(PickerDateRange dateRange) async {
    if (dateRange.startDate == null) return;

    final startDate = dateRange.startDate!;
    final endDate = dateRange.endDate ?? startDate;

    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      // API 호출 파라미터용 날짜 형식
      final formattedStartDate = _formatDate(startDate);
      final formattedEndDate = _formatDate(endDate);

      // 가계부 목록 조회
      final result = await _repository.getHouseholdList(
        startDate: formattedStartDate,
        endDate: formattedEndDate,
      );

      state = state.copyWith(
        isLoading: false,
        transactions: result['allTransactions'],
        groupedTransactions: result['groupedTransactions'],
        totalIncome: result['totalIncome'],
        totalExpenditure: result['totalExpenditure'],
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
    }
  }

  // 거래 삭제
  Future<bool> deleteTransaction(int transactionId) async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      // 삭제할 거래 찾기
      final transactionToDelete = state.transactions.firstWhere(
        (transaction) => transaction.householdPk == transactionId,
        orElse: () => throw Exception('트랜잭션을 찾을 수 없습니다'),
      );

      // 저장소를 통해 거래 삭제
      await _repository.deleteTransaction(transactionId);

      // 삭제 후 상태 업데이트: 해당 거래를 리스트에서 제거
      final updatedTransactions = state.transactions
          .where((transaction) => transaction.householdPk != transactionId)
          .toList();

      // 날짜별 그룹화된 거래도 업데이트
      final updatedGroupedTransactions =
          Map<String, List<Transaction>>.from(state.groupedTransactions);

      // 각 날짜별 그룹에서 해당 거래 제거
      String? dateToRemove;
      updatedGroupedTransactions.forEach((date, transactions) {
        final updatedList = transactions
            .where((transaction) => transaction.householdPk != transactionId)
            .toList();

        updatedGroupedTransactions[date] = updatedList;

        // 해당 날짜의 거래가 모두 삭제되었을 경우 날짜 키 저장
        if (updatedList.isEmpty) {
          dateToRemove = date;
        }
      });

      // 빈 목록이 된 날짜는 제거
      if (dateToRemove != null) {
        updatedGroupedTransactions.remove(dateToRemove);
      }

      // 총액 업데이트
      int updatedIncome = state.totalIncome;
      int updatedExpenditure = state.totalExpenditure;

      // 삭제된 거래가 수입인지 지출인지에 따라 총액 조정
      if (transactionToDelete.classification ==
          TransactionClassification.DEPOSIT) {
        updatedIncome -= transactionToDelete.householdAmount;
      } else if (transactionToDelete.classification ==
          TransactionClassification.WITHDRAWAL) {
        updatedExpenditure -= transactionToDelete.householdAmount;
      }

      state = state.copyWith(
        isLoading: false,
        transactions: updatedTransactions,
        groupedTransactions: updatedGroupedTransactions,
        totalIncome: updatedIncome,
        totalExpenditure: updatedExpenditure,
      );

      return true; // 삭제 성공
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
      return false; // 삭제 실패
    }
  }

  // 가계부 수정
  Future<bool> updateTransaction({
    required int transactionId,
    int? amount,
    String? memo,
    String? exceptedBudgetYn,
    int? detailCategoryPk,
  }) async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      // 저장소를 통해 거래 수정
      final updatedTransaction = await _repository.updateTransaction(
        householdPk: transactionId,
        householdAmount: amount,
        householdMemo: memo,
        exceptedBudgetYn: exceptedBudgetYn,
        householdDetailCategoryPk: detailCategoryPk,
      );

      // 수정된 거래로 상태 업데이트
      final updatedTransactions = state.transactions.map((transaction) {
        if (transaction.householdPk == transactionId) {
          return updatedTransaction;
        }
        return transaction;
      }).toList();

      // 그룹화된 거래도도 업데이트
      final updatedGroupedTransactions =
          Map<String, List<Transaction>>.from(state.groupedTransactions);
      updatedGroupedTransactions.forEach((date, transactions) {
        final index =
            transactions.indexWhere((t) => t.householdPk == transactionId);
        if (index != -1) {
          final updatedList = List<Transaction>.from(transactions);
          updatedList[index] = updatedTransaction;
          updatedGroupedTransactions[date] = updatedList;
        }
      });

      // 총액 업데이트 (금액이 변경된 경우)
      int updatedIncome = state.totalIncome;
      int updatedExpenditure = state.totalExpenditure;

      if (amount != null) {
        // 기존 거래 찾기
        final oldTransaction = state.transactions.firstWhere(
          (t) => t.householdPk == transactionId,
        );

        // 금액 차이 계산
        final amountDiff = amount - oldTransaction.householdAmount;

        // 거래 타입에 따라 총액 조정
        if (oldTransaction.classification ==
            TransactionClassification.DEPOSIT) {
          updatedIncome += amountDiff;
        } else if (oldTransaction.classification ==
            TransactionClassification.WITHDRAWAL) {
          updatedExpenditure += amountDiff;
        }
      }

      state = state.copyWith(
        isLoading: false,
        transactions: updatedTransactions,
        groupedTransactions: updatedGroupedTransactions,
        totalIncome: updatedIncome,
        totalExpenditure: updatedExpenditure,
      );

      return true; // 수정 성공
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
      return false; // 수정 실패
    }
  }
}

final ledgerViewModelProvider =
    StateNotifierProvider<LedgerViewModel, LedgerState>((ref) {
  final repository = ref.watch(ledgerRepositoryProvider);
  return LedgerViewModel(repository);
});
