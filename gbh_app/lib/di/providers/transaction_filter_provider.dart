import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/data/datasources/remote/ledger_api.dart';

// 필터 옵션 열거형
enum TransactionFilterType {
  ALL('전체'),
  INCOME('수입'),
  EXPENSE('지출'),
  TRANSFER('이체');

  final String label;
  const TransactionFilterType(this.label);
}

// 필터 상태를 관리하는 프로바이더
final transactionFilterProvider =
    StateProvider<TransactionFilterType>((ref) => TransactionFilterType.ALL);

// 필터링된 거래 목록 프로바이더
final filteredTransactionsProvider =
    FutureProvider<List<Transaction>>((ref) async {
  // 현재 선택된 필터 가져오기
  final filter = ref.watch(transactionFilterProvider);

  // 현재 선택된 날짜 범위 가져오기
  final datePickerState = ref.watch(datePickerProvider);

  if (datePickerState.selectedRange == null) {
    return [];
  }

  // 날짜 포맷터
  final formatter = DateFormat('yyyyMMdd');
  final startDate = formatter.format(datePickerState.selectedRange!.startDate!);
  final endDate = formatter.format(datePickerState.selectedRange!.endDate ??
      datePickerState.selectedRange!.startDate!);

  // API 인스턴스 가져오기 (의존성 주입 필요)
  final ledgerApi = ref.read(ledgerApiProvider);

  try {
    // 필터에 따라 다른 API 호출
    switch (filter) {
      case TransactionFilterType.ALL:
        // 전체 데이터는 기존 getHouseholdList 사용
        final result = await ledgerApi.getHouseholdList(
            startDate: startDate, endDate: endDate);
        return result['allTransactions'] ?? [];

      case TransactionFilterType.INCOME:
      case TransactionFilterType.EXPENSE:
      case TransactionFilterType.TRANSFER:
        // 분류별 필터링
        final classification = {
          TransactionFilterType.INCOME: 'DEPOSIT',
          TransactionFilterType.EXPENSE: 'WITHDRAWAL',
          TransactionFilterType.TRANSFER: 'TRANSFER'
        }[filter]!;

        final result = await ledgerApi.getHouseholdFilter(
            startDate: startDate,
            endDate: endDate,
            classification: classification);

        // householdList에서 트랜잭션 추출
        final List<Transaction> transactions = [];
        for (var dateGroup in result['householdList'] ?? []) {
          for (var item in dateGroup['list'] ?? []) {
            transactions.add(Transaction.fromJson(item));
          }
        }
        return transactions;
    }
  } catch (e) {
    print('거래 내역 필터링 중 오류 발생: $e');
    return [];
  }
});
