import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_item.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/transaction_form.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/ledger_viewmodel.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/completion_message/completion_message.dart';
import 'package:marshmellow/di/providers/calendar_providers.dart';

class DailyTransactionsContent extends ConsumerStatefulWidget {
  final DateTime date;
  final List<Transaction> transactions;
  final VoidCallback? onTransactionChanged;

  const DailyTransactionsContent({
    Key? key,
    required this.date,
    required this.transactions,
    this.onTransactionChanged,
  }) : super(key: key);

  @override
  ConsumerState<DailyTransactionsContent> createState() =>
      _DailyTransactionsContentState();
}

class _DailyTransactionsContentState
    extends ConsumerState<DailyTransactionsContent> {
  late List<Transaction> _transactions;

  @override
  void initState() {
    super.initState();
    _transactions = List.from(widget.transactions);
  }

  @override
  Widget build(BuildContext context) {
    // 요일 이름 배열 및 날짜 포맷
    final dayNames = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일'];
    final dayName = dayNames[widget.date.weekday - 1];
    final dateString = '${widget.date.month}월 ${widget.date.day}일 $dayName';

    // 숫자 포맷터
    final numberFormat = NumberFormat('#,###', 'ko_KR');

    // 수입/지출 합계 계산
    int income = 0;
    int expense = 0;

    for (var transaction in _transactions) {
      if (transaction.classification == TransactionClassification.DEPOSIT) {
        income += transaction.householdAmount;
      } else if (transaction.classification ==
          TransactionClassification.WITHDRAWAL) {
        expense += transaction.householdAmount;
      }
    }

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // 날짜 정보
        Text(dateString, style: AppTextStyles.modalTitle),

        // 총 건수 및 수입/지출 정보
        const SizedBox(height: 20),
        Row(
          children: [
            Text(
              '총 ${_transactions.length}건',
              style: AppTextStyles.bodyMedium.copyWith(
                color: AppColors.textSecondary,
              ),
            ),
            const Spacer(),
            if (income > 0)
              Text(
                '+ ${numberFormat.format(income)}원',
                style: AppTextStyles.bodyMedium.copyWith(
                  color: AppColors.blueDark,
                ),
              ),
            const SizedBox(width: 12),
            if (expense > 0)
              Text(
                '- ${numberFormat.format(expense)}원',
                style: AppTextStyles.bodyMedium.copyWith(
                  color: AppColors.textPrimary,
                ),
              ),
          ],
        ),

        // 구분선
        const SizedBox(height: 12),
        const Divider(height: 1, thickness: 0.5, color: AppColors.textLight),

        // 거래 목록 또는 빈 상태
        Flexible(
          child: _transactions.isEmpty
              ? Container(
                  alignment: Alignment.center,
                  padding: const EdgeInsets.symmetric(vertical: 40),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Image.asset(
                        'assets/images/characters/char_melong.png',
                        width: 100,
                        height: 100,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        '기록이 없습니다.',
                        style: AppTextStyles.bodyMedium.copyWith(
                          color: AppColors.textSecondary,
                        ),
                      ),
                    ],
                  ),
                )
              : ListView.builder(
                  shrinkWrap: true,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  // 거래 목록 + 내역 추가 항목
                  itemCount: _transactions.length + 1,
                  itemBuilder: (context, index) {
                    // 마지막 항목은 "내역 추가" 버튼
                    if (index == _transactions.length) {
                      return GestureDetector(
                        onTap: () {
                          // 내역 추가 화면으로 이동 또는 추가 모달 표시
                          Navigator.of(context).pop(); // 현재 모달 닫기
                          // 거래 추가 모달 표시
                          showCustomModal(
                            context: context,
                            ref: ref,
                            backgroundColor: AppColors.background,
                            child: TransactionForm(
                              initialDate: widget.date, // 선택한 날짜를 전달
                            ),
                          );
                        },
                        child: Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: Row(
                            children: [
                              Container(
                                width: 40,
                                height: 40,
                                decoration: BoxDecoration(
                                  color: AppColors.whiteLight,
                                  borderRadius: BorderRadius.circular(15),
                                  border: Border.all(
                                    color: AppColors.textLight.withOpacity(0.3),
                                  ),
                                ),
                                child: Icon(
                                  Icons.add,
                                  color: AppColors.textSecondary,
                                  size: 20,
                                ),
                              ),
                              const SizedBox(width: 12),
                              Text(
                                '내역 추가',
                                style: AppTextStyles.bodyMedium.copyWith(
                                  color: AppColors.textSecondary,
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    }

                    final transaction = _transactions[index];
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: TransactionListItem(
                        transaction: transaction,
                        onDelete: (transaction) async {
                          // 삭제 처리
                          final success = await ref
                              .read(ledgerViewModelProvider.notifier)
                              .deleteTransaction(transaction.householdPk);

                          if (context.mounted) {
                            if (success) {
                              // 성공 메시지 표시
                              CompletionMessage.show(context, message: '삭제 완료');

                              // 로컬 상태 업데이트 - 삭제된 항목 제거
                              setState(() {
                                _transactions.removeWhere((t) =>
                                    t.householdPk == transaction.householdPk);
                              });

                              // 거래 목록 새로고침
                              ref.refresh(transactionsProvider);

                              // 캘린더 거래 데이터 새로고침
                              ref.refresh(calendarTransactionsProvider);

                              // 수입/지출 카드 업데이트를 위해 ledgerViewModel 새로고침
                              final datePickerState =
                                  ref.read(datePickerProvider);
                              if (datePickerState.selectedRange != null) {
                                ref
                                    .read(ledgerViewModelProvider.notifier)
                                    .loadHouseholdData(
                                        datePickerState.selectedRange!);
                              }

                              // 삭제 후 콜백 호출
                              if (widget.onTransactionChanged != null) {
                                widget.onTransactionChanged!();
                              }

                              // 모든 트랜잭션이 삭제된 경우 모달 닫기
                              if (_transactions.isEmpty) {
                                Navigator.of(context).pop();
                              }
                            } else {
                              // 실패 메시지 표시
                              CompletionMessage.show(context, message: '삭제 실패');
                            }
                          }
                        },
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }
}

// 거래 모달을 표시하는 함수
void showDailyTransactionsModal({
  required BuildContext context,
  required WidgetRef ref,
  required DateTime date,
  required List<Transaction> transactions,
  VoidCallback? onTransactionChanged,
}) {
  // 거래가 없을 때 더 작은 높이를 사용
  final double modalMaxHeight = transactions.isEmpty
      ? MediaQuery.of(context).size.height * 0.5 // 빈 상태일 때 50%
      : MediaQuery.of(context).size.height * 0.9; // 거래가 있을 때 90%

  showCustomModal(
    context: context,
    ref: ref,
    backgroundColor: AppColors.background,
    padding: const EdgeInsets.all(20),
    maxHeight: modalMaxHeight,
    child: DailyTransactionsContent(
      date: date,
      transactions: transactions,
      onTransactionChanged: onTransactionChanged,
    ),
  );
}
