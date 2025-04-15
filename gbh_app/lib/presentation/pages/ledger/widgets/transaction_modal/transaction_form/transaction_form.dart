import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/constants/icon_path.dart';

// 폼
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/transaction_field.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/type_selector.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/income_form.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/expense_form.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/transfer_form.dart';

// 위젯
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/keyboard/keyboard_modal.dart';
import 'package:marshmellow/presentation/widgets/completion_message/completion_message.dart';

// 가계부 관련 ViewModel
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/ledger_viewmodel.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:marshmellow/di/providers/transaction_filter_provider.dart';

// TransactionForm을 ConsumerStatefulWidget으로 변환
class TransactionForm extends ConsumerStatefulWidget {
  final DateTime? initialDate;

  const TransactionForm({
    super.key,
    this.initialDate,
  });

  @override
  ConsumerState<TransactionForm> createState() => _TransactionFormState();
}

class _TransactionFormState extends ConsumerState<TransactionForm> {
  String _selectedType = '지출'; // 기본값을 지출로 설정
  String _amount = ''; // 금액 상태 빈 문자열로 변경
  bool _isSaving = false; // 저장 중 상태

  // 폼 레퍼런스 관리
  final _incomeFormKey = GlobalKey<IncomeFormState>();
  final _expenseFormKey = GlobalKey<ExpenseFormState>();
  final _transferFormKey = GlobalKey<TransferFormState>();

  // 선택된 타입에 따라 다른 폼을 반환하는 메서드 (수정)
  Widget _getFormByType() {
    final initialDate = widget.initialDate;

    switch (_selectedType) {
      case '수입':
        return IncomeForm(
          key: _incomeFormKey,
          initialDate: initialDate,
        );
      case '지출':
        return ExpenseForm(
          key: _expenseFormKey,
          initialDate: initialDate,
        );
      case '이체':
        return TransferForm(
          key: _transferFormKey,
          initialDate: initialDate,
        );
      default:
        return ExpenseForm(
          key: _expenseFormKey,
          initialDate: initialDate,
        );
    }
  }

  // 금액을 포맷팅하는 메서드
  String _formatAmount(String amount) {
    String numbers = amount.replaceAll(RegExp(r'[^0-9]'), '');
    if (numbers.isEmpty) return '0';

    final formatted = int.parse(numbers)
        .toString()
        .split('')
        .reversed
        .join('')
        .replaceAllMapped(
          RegExp(r'.{3}'),
          (match) => '${match.group(0)},',
        )
        .split('')
        .reversed
        .join('')
        .replaceAll(RegExp(r'^,'), '');

    return formatted;
  }

  // 계산기 키보드를 표시하는 메서드
  void _showCalculatorKeyboard() {
    KeyboardModal.showCalculatorKeyboard(
      context: context,
      initialValue: _amount.replaceAll(',', ''),
      onValueChanged: (value) {
        setState(() {
          _amount = value;
        });
      },
    );
  }

  // 가계부 내역 저장 메서드
  Future<void> _saveTransaction() async {
    // 금액이 0이거나 비어있으면 저장하지 않음
    if (_amount.isEmpty || _amount == '0') {
      CompletionMessage.show(context, message: '금액 입력');
      return;
    }

    // 저장 중 상태로 변경
    setState(() {
      _isSaving = true;
    });

    try {
      // 금액 파싱
      final int amountValue = int.parse(_amount.replaceAll(',', ''));

      // 가계부 분류 변환
      String householdClassification;
      switch (_selectedType) {
        case '수입':
          householdClassification = 'DEPOSIT';
          break;
        case '지출':
          householdClassification = 'WITHDRAWAL';
          break;
        case '이체':
          householdClassification = 'TRANSFER';
          break;
        default:
          householdClassification = 'WITHDRAWAL';
      }

      // 현재 날짜와 시간 (기본값)
      DateTime now = DateTime.now();
      String tradeDate = DateFormat('yyyyMMdd').format(now);
      String tradeTime = DateFormat('HHmm').format(now);

      // 폼 데이터 수집
      Map<String, dynamic> formData = {};

      switch (_selectedType) {
        case '수입':
          if (_incomeFormKey.currentState != null) {
            formData = _incomeFormKey.currentState!.getFormData();
            if (formData['date'] != null) {
              final DateTime selectedDate = formData['date'];
              tradeDate = DateFormat('yyyyMMdd').format(selectedDate);
              tradeTime = DateFormat('HHmm').format(selectedDate);
            }
          }
          break;
        case '지출':
          if (_expenseFormKey.currentState != null) {
            formData = _expenseFormKey.currentState!.getFormData();
            if (formData['date'] != null) {
              final DateTime selectedDate = formData['date'];
              tradeDate = DateFormat('yyyyMMdd').format(selectedDate);
              tradeTime = DateFormat('HHmm').format(selectedDate);
            }
          }
          break;
        case '이체':
          if (_transferFormKey.currentState != null) {
            formData = _transferFormKey.currentState!.getFormData();
            if (formData['date'] != null) {
              final DateTime selectedDate = formData['date'];
              tradeDate = DateFormat('yyyyMMdd').format(selectedDate);
              tradeTime = DateFormat('HHmm').format(selectedDate);
            }
          }
          break;
      }

      // 필수 필드 검증
      if (formData['tradeName'] == null ||
          formData['paymentMethod'] == null ||
          formData['categoryPk'] == null) {
        CompletionMessage.show(context, message: '모든항목입력');
        setState(() {
          _isSaving = false;
        });
        return;
      }

      // LedgerRepository 인스턴스 가져오기
      final repository = ref.read(ledgerRepositoryProvider);

      // API 호출
      await repository.createHousehold(
        tradeName: formData['tradeName'],
        tradeDate: tradeDate,
        tradeTime: tradeTime,
        householdAmount: amountValue,
        householdMemo: formData['memo'],
        paymentMethod: formData['paymentMethod'],
        exceptedBudgetYn: formData['exceptedBudgetYn'] ?? 'N',
        householdClassification: householdClassification,
        householdDetailCategoryPk: formData['categoryPk'],
      );

      // 저장 성공
      CompletionMessage.show(context, message: '저장 완료');

      // 트랜잭션 목록 및 필터링된 트랜잭션 새로고침
      ref.invalidate(transactionsProvider);
      ref.invalidate(filteredTransactionsProvider);

      // 월별 통계 새로고침
      final datePickerState = ref.read(datePickerProvider);
      if (datePickerState.selectedRange != null) {
        ref
            .read(ledgerViewModelProvider.notifier)
            .loadHouseholdData(datePickerState.selectedRange!);
      }

      // 모달 닫기
      if (context.mounted) {
        Navigator.of(context).pop();
      }
    } catch (e) {
      print('가계부 내역 저장 중 오류: $e');
      if (context.mounted) {
        CompletionMessage.show(context, message: '저장 오류');
      }
    } finally {
      // 저장 중 상태 해제
      if (mounted) {
        setState(() {
          _isSaving = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final ledgerState = ref.watch(ledgerViewModelProvider);

    return Stack(
      children: [
        // 콘텐츠 영역
        Padding(
          padding: const EdgeInsets.only(bottom: 70), // 하단 버튼 높이만큼 패딩 추가
          child: SingleChildScrollView(
            child: Column(
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 20),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.start,
                    children: [
                      Text(
                        _formatAmount(_amount),
                        style: AppTextStyles.modalMoneyTitle,
                      ),
                      Text(
                        ' 원',
                        style: AppTextStyles.bodyMedium,
                      ),
                      IconButton(
                        onPressed: _showCalculatorKeyboard, // 계산기 키보드 호출
                        icon: SvgPicture.asset(IconPath.pencilSimple),
                      ),
                      const SizedBox(width: 10),
                    ],
                  ),
                ),
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
                    ),
                  ),
                ),
                // 선택된 타입에 따른 폼 표시
                _getFormByType(),
                const SizedBox(height: 10),
              ],
            ),
          ),
        ),

        // 하단 고정 버튼
        Positioned(
          left: 0,
          right: 0,
          bottom: 0,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: Theme.of(context).scaffoldBackgroundColor,
            ),
            child: Button(
              text: '저장하기',
              height: 50,
              width: screenWidth * 0.9,
              isDisabled: _isSaving || ledgerState.isLoading,
              onPressed:
                  _isSaving || ledgerState.isLoading ? null : _saveTransaction,
            ),
          ),
        ),
      ],
    );
  }
}
