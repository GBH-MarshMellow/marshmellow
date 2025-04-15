// presentation/pages/finance/widgets/financial_section_widget.dart
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart'; // 추가
import 'package:marshmellow/core/theme/app_colors.dart'; // 추가
import 'package:flutter_riverpod/flutter_riverpod.dart'; // 추가
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';


class FinancialSectionWidget extends ConsumerWidget {
  
  final String title;
  final int totalAmount;
  final List<Widget> itemList;
  final bool isEmpty;
  final String emptyMessage;

  const FinancialSectionWidget({
    Key? key,
    required this.title,
    required this.totalAmount,
    required this.itemList,
    this.isEmpty = false,
    this.emptyMessage = '등록된 정보가 없습니다.',
  }) : super(key: key);

  // 숫자 포맷팅 함수 (천 단위 구분)
  String formatAmount(int amount) {
    final formatter = NumberFormat('#,###');
    return formatter.format(amount);
  }

@override
Widget build(BuildContext context, WidgetRef ref) {
  final isHidden = ref.watch(isFinanceHideProvider);
  return Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      const SizedBox(height: 20),
      Row(
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(left: 8.0),
            child: Text(
              title,
              style: AppTextStyles.bodySmall.copyWith(color: AppColors.blackLight),
            ),
          ),
          const SizedBox(width: 6),
          if (!isEmpty)
            Padding(
              padding: const EdgeInsets.only(right: 8.0),
              child: Text(
                isHidden ? '금액보기' : '${formatAmount(totalAmount)}원',
                style: AppTextStyles.bodySmall.copyWith(color: AppColors.blackLight),
              ),
            ),
        ],
      ),
      if (isEmpty)
        Padding(
          padding: const EdgeInsets.only(left: 8.0),
          child: Text(emptyMessage),
        )
      else
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 4),
            ...itemList,
          ],
        ),
      const SizedBox(height: 16),
    ],
  );
}
}