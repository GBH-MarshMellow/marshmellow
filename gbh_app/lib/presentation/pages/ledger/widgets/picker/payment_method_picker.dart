// lib/presentation/pages/ledger/widgets/picker/payment_method_picker.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/payment_method.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/payment_method_viewmodel.dart';

// 결제수단 선택기 위젯
class PaymentMethodPicker extends ConsumerStatefulWidget {
  final String transactionType; // 'income', 'expense', 'transfer'
  final Function(PaymentMethod) onPaymentMethodSelected;
  final String? title;

  const PaymentMethodPicker({
    Key? key,
    required this.transactionType,
    required this.onPaymentMethodSelected,
    this.title,
  }) : super(key: key);

  @override
  ConsumerState<PaymentMethodPicker> createState() =>
      _PaymentMethodPickerState();
}

class _PaymentMethodPickerState extends ConsumerState<PaymentMethodPicker> {
  bool _isLoading = true;
  String _errorMessage = '';

  @override
  void initState() {
    super.initState();
    // Future.microtask를 사용하여 위젯 빌드 완료 후 로딩
    Future.microtask(() => _loadPaymentMethods());
  }

  Future<void> _loadPaymentMethods() async {
    setState(() {
      _isLoading = true;
      _errorMessage = '';
    });

    try {
      await ref
          .read(paymentMethodViewModelProvider.notifier)
          .loadPaymentMethods();
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _errorMessage = '결제수단 목록을 불러오는데 실패했습니다: $e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // 결제수단 상태 구독
    final paymentMethodState = ref.watch(paymentMethodViewModelProvider);

    // 트랜잭션 타입에 따라 필터링된 결제수단 목록
    List<PaymentMethod> filteredMethods = [];

    if (!_isLoading &&
        _errorMessage.isEmpty &&
        paymentMethodState.paymentMethods.isNotEmpty) {
      switch (widget.transactionType.toLowerCase()) {
        case 'income':
          // 수입: 계좌만 표시
          filteredMethods = paymentMethodState.paymentMethods
              .where((method) => method.paymentType == 'ACCOUNT')
              .toList();
          break;
        case 'expense':
          // 지출: 현금, 카드, 계좌 모두 표시
          filteredMethods = paymentMethodState.paymentMethods;
          break;
        case 'transfer':
          // 이체: 계좌만 표시
          filteredMethods = paymentMethodState.paymentMethods
              .where((method) => method.paymentType == 'ACCOUNT')
              .toList();
          break;
        default:
          filteredMethods = paymentMethodState.paymentMethods;
      }
    }

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(10),
          topRight: Radius.circular(10),
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // 핸들바
          Center(
            child: Container(
              margin: const EdgeInsets.only(top: 12, bottom: 20),
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.grey.shade300,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),

          // 로딩 중이거나 에러 발생 시
          if (_isLoading)
            Container(
              padding: const EdgeInsets.symmetric(vertical: 30),
              child: const Center(
                child: CircularProgressIndicator(),
              ),
            )
          else if (_errorMessage.isNotEmpty)
            Container(
              padding: const EdgeInsets.symmetric(vertical: 30),
              child: Center(
                child: Text(
                  _errorMessage,
                  style: AppTextStyles.bodyMedium
                      .copyWith(color: AppColors.warnning),
                ),
              ),
            )
          // 결제수단 목록
          else if (filteredMethods.isEmpty &&
              widget.transactionType.toLowerCase() == 'transfer')
            Container(
              padding: const EdgeInsets.symmetric(vertical: 30),
              child: Center(
                child: Text(
                  '선택 가능한 결제수단이 없습니다.',
                  style: AppTextStyles.bodyMedium,
                ),
              ),
            )
          else
            ConstrainedBox(
              constraints: BoxConstraints(
                maxHeight: MediaQuery.of(context).size.height * 0.8, // 높이 제한
              ),
              child: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min, // 내용에 맞게 크기 조절
                  children: [
                    // 현금 항목 (이체가 아닌 경우에만 표시)
                    if (widget.transactionType.toLowerCase() != 'transfer')
                      _buildPaymentMethodItem(
                        name: '현금',
                        onTap: () {
                          final cashMethod = PaymentMethod(
                            bankCode: 'CASH',
                            bankName: '현금',
                            paymentType: 'CASH',
                            paymentMethod: '현금',
                          );
                          widget.onPaymentMethodSelected(cashMethod);
                          Navigator.of(context).pop();
                        },
                      ),

                    // 필터링된 결제수단 목록
                    ...filteredMethods.map((method) => _buildPaymentMethodItem(
                          name: method.paymentMethod,
                          onTap: () {
                            widget.onPaymentMethodSelected(method);
                            Navigator.of(context).pop();
                          },
                        )),
                  ],
                ),
              ),
            ),

          // 하단 여백
          const SizedBox(height: 30),
        ],
      ),
    );
  }

  // 결제수단 항목 위젯 - 아이콘 없이 텍스트만 표시
  Widget _buildPaymentMethodItem({
    required String name,
    required VoidCallback onTap,
  }) {
    return Column(
      children: [
        InkWell(
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 15),
            child: SizedBox(
              width: double.infinity,
              child: Text(
                name,
                style: AppTextStyles.bodyMedium,
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// 결제수단 선택 모달 표시 함수
Future<void> showPaymentMethodPickerModal(
  BuildContext context, {
  required String transactionType, // 'income', 'expense', 'transfer'
  required Function(PaymentMethod) onPaymentMethodSelected,
  String? title,
}) {
  return showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (context) => PaymentMethodPicker(
      transactionType: transactionType,
      onPaymentMethodSelected: onPaymentMethodSelected,
      title: title,
    ),
  );
}
