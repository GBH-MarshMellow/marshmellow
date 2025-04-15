import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/di/providers/my/salary_provider.dart';

// 월급일 프로바이더 (초기값은 1로 설정)
final paydayProvider = StateProvider<int>((ref) => 1);

// 월급일 조회 및 업데이트 함수를 가진 프로바이더
final paydayFetchProvider = FutureProvider<int>((ref) async {
  final salaryRepository = ref.watch(mySalaryRepositoryProvider);
  try {
    final payday = await salaryRepository.getSalaryDay();
    // 결과를 paydayProvider에 저장
    ref.read(paydayProvider.notifier).state = payday;
    return payday;
  } catch (e) {
    // 에러 발생 시 기본값 1 유지
    return 1;
  }
});

// 캘린더 기간 프로바이더 - 월급일 기준
final calendarPeriodProvider =
    StateProvider<(DateTime start, DateTime end)>((ref) {
  final now = DateTime.now();
  int payday = ref.watch(paydayProvider); // 월급일 프로바이더에서 값 가져오기

  DateTime startDay;
  // 현재 날짜가 월급일 이전이면 전 달부터, 아니면 현재 달부터
  if (now.day < payday) {
    startDay = DateTime(now.year, now.month - 1, payday);
  } else {
    startDay = DateTime(now.year, now.month, payday);
  }

  // 다음 월급일 계산
  DateTime endDay;
  if (startDay.month == 12) {
    endDay = DateTime(startDay.year + 1, 1, payday)
        .subtract(const Duration(days: 1));
  } else {
    endDay = DateTime(startDay.year, startDay.month + 1, payday)
        .subtract(const Duration(days: 1));
  }

  return (startDay, endDay);
});

// 캘린더 트랜잭션 프로바이더
final calendarTransactionsProvider =
    FutureProvider<List<Transaction>>((ref) async {
  // transactionsProvider를 감시하여 변경 시 자동으로 업데이트
  await ref.watch(transactionsProvider.future);

  final repository = ref.watch(transactionRepositoryProvider);
  final period = ref.watch(calendarPeriodProvider);

  return repository.getTransactions(
    startDate: period.$1,
    endDate: period.$2,
  );
});
