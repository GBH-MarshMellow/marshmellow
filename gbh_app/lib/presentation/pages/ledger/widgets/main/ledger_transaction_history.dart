import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/category/transaction_category.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_item.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/ledger_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/completion_message/completion_message.dart';
import 'package:flutter_slidable/flutter_slidable.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:marshmellow/di/providers/transaction_filter_provider.dart';

class LedgerTransactionHistory extends ConsumerStatefulWidget {
  const LedgerTransactionHistory({super.key});

  @override
  ConsumerState<LedgerTransactionHistory> createState() =>
      _LedgerTransactionHistoryState();
}

class _LedgerTransactionHistoryState
    extends ConsumerState<LedgerTransactionHistory> {
  final _numberFormat = NumberFormat('#,###', 'ko_KR');
  @override
  void dispose() {
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 필터링된 트랜잭션 데이터 구독
    final filteredTransactionsAsync = ref.watch(filteredTransactionsProvider);

    return filteredTransactionsAsync.when(
      data: (transactions) {
        if (transactions.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Image.asset(
                  'assets/images/characters/char_melong.png',
                  width: 150,
                  height: 150,
                ),
                Text('기록이 없습니다.'),
              ],
            ),
          );
        }

        // 트랜잭션을 날짜별로 그룹화
        final repository = ref.watch(transactionRepositoryProvider);
        final grouped = repository.groupTransactionsByDate(transactions);

        // 날짜 목록을 내림차순으로 정렬 (최신순)
        final dates = grouped.keys.toList()..sort((a, b) => b.compareTo(a));

        return ListView.builder(
          physics: const AlwaysScrollableScrollPhysics(),
          shrinkWrap: true,
          itemCount: dates.length,
          itemBuilder: (context, index) {
            final date = dates[index];
            final items = grouped[date]!;

            // 해당 날짜의 수입/지출 합계
            double dayIncome = 0;
            double dayExpense = 0;

            for (var transaction in transactions) {
              if (transaction.dateTime.year == date.year &&
                  transaction.dateTime.month == date.month &&
                  transaction.dateTime.day == date.day) {
                if (transaction.classification ==
                    TransactionClassification.DEPOSIT) {
                  dayIncome += transaction.householdAmount.toDouble();
                } else if (transaction.classification ==
                    TransactionClassification.WITHDRAWAL) {
                  dayExpense += transaction.householdAmount.toDouble();
                }
              }
            }

            // 요일 이름 (월요일, 화요일, ...)
            final dayNames = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'];
            final dayName = dayNames[date.weekday - 1];

            return Column(
              key: ValueKey('date-group-${date.toString()}'),
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // 날짜 헤더
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 5.0),
                  child: Row(
                    children: [
                      Text(
                        '${date.month}월 ${date.day}일 $dayName',
                        style: AppTextStyles.bodyMedium.copyWith(
                          fontWeight: FontWeight.w300,
                          color: AppColors.textPrimary,
                        ),
                      ),
                      const Spacer(),
                      // 수입 합계
                      if (dayIncome > 0)
                        Text(
                          '+ ${_numberFormat.format(dayIncome)}원',
                          style: AppTextStyles.bodySmall.copyWith(
                            color: AppColors.blueDark,
                            fontWeight: FontWeight.w400,
                          ),
                        ),
                      const SizedBox(width: 10),
                      // 지출 합계
                      if (dayExpense > 0)
                        Text(
                          '- ${_numberFormat.format(dayExpense)}원',
                          style: AppTextStyles.bodySmall.copyWith(
                            color: AppColors.textSecondary,
                            fontWeight: FontWeight.w400,
                          ),
                        ),
                    ],
                  ),
                ),
                // 날짜 바로 아래에 Divider 추가
                const Divider(
                    height: 1, thickness: 0.5, color: AppColors.textSecondary),

                // 해당 날짜의 거래 목록
                const SizedBox(height: 10),
                ...items
                    .map((item) => Padding(
                          key: ValueKey('transaction-item-${item.householdPk}'),
                          padding: const EdgeInsets.only(bottom: 12),
                          child: TransactionListItem(
                            transaction: item,
                            onDelete: (transaction) async {
                              // 삭제 처리
                              final success = await ref
                                  .read(ledgerViewModelProvider.notifier)
                                  .deleteTransaction(transaction.householdPk);

                              if (context.mounted) {
                                if (success) {
                                  // 성공 메시지 표시
                                  CompletionMessage.show(context,
                                      message: '삭제 완료');

                                  // 필터된 트랜잭션 다시 로드
                                  ref.refresh(filteredTransactionsProvider);

                                  // 또한 ledgerViewModel 새로고침 (수입/지출 카드 업데이트를 위해)
                                  final datePickerState =
                                      ref.read(datePickerProvider);
                                  if (datePickerState.selectedRange != null) {
                                    ref
                                        .read(ledgerViewModelProvider.notifier)
                                        .loadHouseholdData(
                                            datePickerState.selectedRange!);
                                  }
                                } else {
                                  // 실패 메시지 표시
                                  CompletionMessage.show(context,
                                      message: '삭제 실패');
                                }
                              }
                            },
                          ),
                        ))
                    .toList(),
              ],
            );
          },
        );
      },
      loading: () => Center(
        child: Lottie.asset(
          'assets/images/loading/loading_simple.json',
          width: 140,
          height: 140,
          fit: BoxFit.contain,
        ),
      ),
      error: (error, stack) => Center(
        child: Text('오류가 발생했습니다: $error'),
      ),
    );
  }
}
