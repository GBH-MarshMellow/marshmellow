import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/data/models/ledger/category/transaction_category.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/transaction_field.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/type_selector.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/completion_message/completion_message.dart';
import 'package:marshmellow/di/providers/calendar_providers.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/ledger_viewmodel.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/presentation/widgets/keyboard/keyboard_modal.dart';
import 'package:marshmellow/data/models/ledger/category/category_mapping.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/transfer_direction_picker.dart';
import 'package:marshmellow/di/providers/transaction_filter_provider.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_detail/category_expense_list_page.dart';
import 'package:marshmellow/presentation/viewmodels/budget/budget_viewmodel.dart';

// 기존 폼들 import
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/expense_form.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/income_form.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/transfer_form.dart';

// 상세 정보 프로바이더 추가
final transactionDetailProvider =
    FutureProvider.family<Transaction, int>((ref, householdPk) async {
  final repository = ref.watch(ledgerRepositoryProvider);
  return repository.getHouseholdDetail(householdPk);
});

class TransactionDetailModal extends ConsumerStatefulWidget {
  final int householdPk;
  final bool readOnly;

  const TransactionDetailModal({
    Key? key,
    required this.householdPk,
    this.readOnly = false,
  }) : super(key: key);

  @override
  ConsumerState<TransactionDetailModal> createState() =>
      _TransactionDetailModalState();
}

class _TransactionDetailModalState
    extends ConsumerState<TransactionDetailModal> {
  String _selectedType = '지출';
  // 수정할 데이터를 저장할 변수들
  int? _updatedAmount;
  String? _updatedMemo;
  String? _updatedExceptedBudgetYn;
  int? _updatedDetailCategoryPk;

  @override
  Widget build(BuildContext context) {
    final transactionDetailAsync =
        ref.watch(transactionDetailProvider(widget.householdPk));
    final numberFormat = NumberFormat('#,###', 'ko_KR');

    return transactionDetailAsync.when(
      data: (transaction) {
        // 트랜잭션 타입에 따라 선택된 타입 업데이트
        WidgetsBinding.instance.addPostFrameCallback((_) {
          String typeText = '지출';
          switch (transaction.type) {
            case TransactionType.deposit:
              typeText = '수입';
              break;
            case TransactionType.withdrawal:
              typeText = '지출';
              break;
            case TransactionType.transfer:
              typeText = '이체';
              break;
          }

          if (_selectedType != typeText) {
            setState(() {
              _selectedType = typeText;
            });
          }
        });
        return Stack(
          children: [
            // 콘텐츠 영역
            Padding(
              padding: EdgeInsets.only(bottom: widget.readOnly ? 0 : 70),
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    // 금액 표시
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 20),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.start,
                        children: [
                          Text(
                            '${numberFormat.format(_updatedAmount ?? transaction.amount)}',
                            style: AppTextStyles.modalMoneyTitle.copyWith(
                              color: AppColors.textPrimary,
                            ),
                          ),
                          Text(
                            ' 원',
                            style: AppTextStyles.bodyMedium,
                          ),
                          if (!widget.readOnly)
                            IconButton(
                              onPressed: () {
                                // 계산기 키보드 열기
                                final initialValue =
                                    (_updatedAmount ?? transaction.amount)
                                        .toString();
                                // 초기값이 .0으로 끝나는 경우 소수점 제거
                                final formattedInitialValue =
                                    initialValue.endsWith('.0')
                                        ? initialValue.substring(
                                            0, initialValue.length - 2)
                                        : initialValue;

                                KeyboardModal.showCalculatorKeyboard(
                                  context: context,
                                  initialValue: formattedInitialValue,
                                  onValueChanged: (value) {
                                    setState(() {
                                      // 소수점 제거: 값이 정수인 경우 .0 제거
                                      if (value.endsWith('.0')) {
                                        value = value.substring(
                                            0, value.length - 2);
                                      }
                                      _updatedAmount = int.tryParse(value) ??
                                          transaction.amount.toInt();
                                    });
                                  },
                                );
                              },
                              icon: SvgPicture.asset(IconPath.pencilSimple),
                            ),
                          const SizedBox(width: 10),
                        ],
                      ),
                    ),

                    // 분류 선택기
                    TransactionField(
                      label: '분류',
                      trailing: SizedBox(
                        width: 200,
                        child: TypeSelector(
                          selectedType: _selectedType,
                          onTypeSelected: (type) {
                            setState(() {
                              _selectedType = type;
                            });
                          },
                          enabled: false,
                        ),
                      ),
                    ),

                    // 선택된 타입에 따른 폼 표시 (기존 폼 사용)
                    _getFormByType(transaction),
                  ],
                ),
              ),
            ),

            // 하단 고정 버튼 (readOnly가 아닐 때만 표시)
            if (!widget.readOnly)
              Positioned(
                left: 0,
                right: 0,
                bottom: 0,
                child: Center(
                    child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Button(
                      text: '삭제',
                      width: MediaQuery.of(context).size.width * 0.43,
                      onPressed: () async {
                        // 삭제 처리
                        final success = await ref
                            .read(ledgerViewModelProvider.notifier)
                            .deleteTransaction(widget.householdPk);

                        if (context.mounted) {
                          if (success) {
                            // 성공 메시지 표시
                            CompletionMessage.show(context, message: '삭제 완료');
                            // 모달 닫기
                            Navigator.of(context).pop();

                            // 트랜잭션 목록 새로고침
                            ref.refresh(transactionsProvider);

                            // 캘린더 데이터 새로고침
                            ref.refresh(calendarTransactionsProvider);

                            // 카테고리 트랜잭션 데이터 새로고침
                            ref.invalidate(categoryTransactionsProvider);

                            // ledgerViewModel 새로고침 (수입/지출 카드 업데이트)
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
                            CompletionMessage.show(context, message: '삭제 실패');
                          }
                        }
                      },
                    ),
                    const SizedBox(width: 10), // 버튼 사이 간격
                    Button(
                      text: '수정',
                      width: MediaQuery.of(context).size.width * 0.43,
                      onPressed: () async {
                        // 변경된 값만 업데이트하기 위한 매개변수 준비
                        Map<String, dynamic> updateParams = {};
                        updateParams['transactionId'] = transaction.householdPk;

                        if (_updatedAmount != null) {
                          updateParams['amount'] = _updatedAmount;
                        }

                        if (_updatedMemo != null) {
                          updateParams['memo'] = _updatedMemo;
                        }

                        if (_updatedExceptedBudgetYn != null) {
                          updateParams['exceptedBudgetYn'] =
                              _updatedExceptedBudgetYn;
                        }

                        // API 문서에 따르면 카테고리 고유번호는 필수 필드
                        int? categoryPkNullable = _updatedDetailCategoryPk ??
                            CategoryPkMapping.getPkFromCategory(
                                expenseCategory: transaction.type == TransactionType.withdrawal
                                    ? ref
                                        .read(ledgerRepositoryProvider)
                                        .getWithdrawalCategoryByName(
                                            transaction.householdCategory)
                                    : null,
                                incomeCategory: transaction.type == TransactionType.deposit
                                    ? ref
                                        .read(ledgerRepositoryProvider)
                                        .getDepositCategoryByName(
                                            transaction.householdCategory)
                                    : null,
                                transferCategory: transaction.type == TransactionType.transfer
                                    ? ref
                                        .read(ledgerRepositoryProvider)
                                        .getTransferCategoryByName(
                                            transaction.householdCategory)
                                    : null,
                                transferDirection: transaction.type == TransactionType.transfer
                                    ? TransferDirection.withdrawal
                                    : null);

                        // 디폴트 값 설정 (기타 카테고리 사용)
                        int categoryPk = categoryPkNullable ??
                            (transaction.type == TransactionType.withdrawal
                                ? 121
                                : // 기타 지출
                                transaction.type == TransactionType.deposit
                                    ? 138
                                    : // 기타 수입
                                    130); // 기타 이체 (출금)

                        updateParams['detailCategoryPk'] = categoryPk;

                        // API 호출
                        final success = await ref
                            .read(ledgerViewModelProvider.notifier)
                            .updateTransaction(
                              transactionId: transaction.householdPk,
                              amount: _updatedAmount,
                              memo: _updatedMemo,
                              exceptedBudgetYn: _updatedExceptedBudgetYn,
                              detailCategoryPk: categoryPk,
                            );

                        if (context.mounted) {
                          if (success) {
                            // 성공하면 먼저 캐시 갱신
                            ref.invalidate(transactionDetailProvider(
                                transaction.householdPk));
                            ref.invalidate(transactionsProvider);
                            ref.invalidate(calendarTransactionsProvider);
                            ref.invalidate(filteredTransactionsProvider);
                            ref.invalidate(categoryTransactionsProvider);
                            ref.invalidate(budgetProvider);

                            // datePickerState와 관련된 데이터도 갱신
                            final datePickerState =
                                ref.read(datePickerProvider);
                            if (datePickerState.selectedRange != null) {
                              ref
                                  .read(ledgerViewModelProvider.notifier)
                                  .loadHouseholdData(
                                      datePickerState.selectedRange!);
                            }

                            // 그 다음 화면 닫기
                            Navigator.of(context).pop();

                            // 마지막으로 성공 메시지 표시
                            CompletionMessage.show(context, message: '수정 완료');
                          } else {
                            // 실패 메시지 표시 (여기서는 화면을 닫지 않음)
                            CompletionMessage.show(context, message: '수정 실패');
                          }
                        }
                      },
                    ),
                  ],
                )),
              ),
          ],
        );
      },
      loading: () => const Center(
        child: CircularProgressIndicator(),
      ),
      error: (error, stack) => Center(
        child: Text('상세 정보를 불러오는 중 오류가 발생했습니다: $error'),
      ),
    );
  }

  Widget _getFormByType(Transaction transaction) {
    switch (_selectedType) {
      case '수입':
        return IncomeForm(
          initialData: transaction,
          onDataChanged: (amount, memo, categoryPk) {
            _updatedAmount = amount;
            _updatedMemo = memo;
            _updatedDetailCategoryPk = categoryPk;
          },
          readOnly: true,
        );
      case '지출':
        return ExpenseForm(
          initialData: transaction,
          onDataChanged: (amount, memo, exceptedBudgetYn, categoryPk) {
            _updatedAmount = amount;
            _updatedMemo = memo;
            _updatedExceptedBudgetYn = exceptedBudgetYn;
            _updatedDetailCategoryPk = categoryPk;
          },
          readOnly: widget.readOnly,
        );
      case '이체':
        return TransferForm(
          initialData: transaction,
          onDataChanged: (amount, memo, categoryPk) {
            _updatedAmount = amount;
            _updatedMemo = memo;
            _updatedDetailCategoryPk = categoryPk;
          },
          readOnly: true,
        );
      default:
        return ExpenseForm(
          initialData: transaction,
          onDataChanged: (amount, memo, exceptedBudgetYn, categoryPk) {
            _updatedAmount = amount;
            _updatedMemo = memo;
            _updatedExceptedBudgetYn = exceptedBudgetYn;
            _updatedDetailCategoryPk = categoryPk;
          },
          readOnly: true,
        );
    }
  }
}
