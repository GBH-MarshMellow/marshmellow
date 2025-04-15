import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/data/repositories/ledger/ledger_repository.dart';
import 'package:syncfusion_flutter_datepicker/datepicker.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:marshmellow/data/datasources/remote/ledger_api.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'package:marshmellow/di/providers/calendar_providers.dart';

// 가계부 API 프로바이더
final ledgerApiProvider = Provider<LedgerApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return LedgerApi(apiClient);
});

// 가계부 저장소 프로바이더
final ledgerRepositoryProvider = Provider<LedgerRepository>((ref) {
  final ledgerApi = ref.watch(ledgerApiProvider);
  return LedgerRepository(ledgerApi);
});

// 거래 저장소 프로바이더 (ledgerRepositoryProvider와 동일하지만 이름을 다르게 사용)
final transactionRepositoryProvider = Provider<LedgerRepository>((ref) {
  return ref.watch(ledgerRepositoryProvider);
});

// 현재 선택된 날짜 범위를 확인하는 프로바이더
final selectedDateRangeProvider = Provider<PickerDateRange>((ref) {
  final datePickerState = ref.watch(datePickerProvider);
  final payday = ref.watch(paydayProvider); // 월급일 정보 추가

  // 선택된 날짜 범위가 없으면 월급일 기준으로 계산
  if (datePickerState.selectedRange == null ||
      datePickerState.selectedRange!.startDate == null) {
    final now = DateTime.now();
    DateTime startDate;
    DateTime endDate;

    // 현재 날짜가 월급일 이전이면 전 달의 월급일부터
    if (now.day < payday) {
      startDate = DateTime(now.year, now.month - 1, payday);
      endDate = DateTime(now.year, now.month, payday - 1);
    } else {
      // 현재 날짜가 월급일 이후면 현재 달의 월급일부터
      startDate = DateTime(now.year, now.month, payday);

      // 다음 달의 월급일 이전 날까지
      if (startDate.month == 12) {
        endDate = DateTime(startDate.year + 1, 1, payday - 1);
      } else {
        endDate = DateTime(now.year, now.month + 1, payday - 1);
      }
    }

    return PickerDateRange(startDate, endDate);
  }

  return datePickerState.selectedRange!;
});

// 거래 목록 비동기 프로바이더
final transactionsProvider = FutureProvider<List<Transaction>>((ref) async {
  final repository = ref.watch(transactionRepositoryProvider);
  final dateRange = ref.watch(selectedDateRangeProvider);

  final startDate = dateRange.startDate!;
  final endDate = dateRange.endDate ?? startDate;

  final result = await repository.getHouseholdList(
    startDate: _formatDate(startDate),
    endDate: _formatDate(endDate),
  );

  return result['allTransactions'] as List<Transaction>;
});

// 날짜 포맷 변환 (yyyyMMdd)
String _formatDate(DateTime date) {
  return '${date.year}${date.month.toString().padLeft(2, '0')}${date.day.toString().padLeft(2, '0')}';
}

// 선택된 트랜잭션 ID 상태 프로바이더
final selectedTransactionIdProvider = StateProvider<int?>((ref) => null);

// 거래 상세 정보 비동기 프로바이더
final transactionDetailProvider =
    FutureProvider.family<Transaction, int>((ref, householdPk) async {
  final repository = ref.watch(ledgerRepositoryProvider);
  return repository.getHouseholdDetail(householdPk);
});

// 날짜 범위 초기화 수행 여부를 추적하는 프로바이더
final dateRangeInitializedProvider = StateProvider<bool>((ref) => false);
