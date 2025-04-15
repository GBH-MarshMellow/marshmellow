import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_item.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';

class CategoryBottomSheetContent extends ConsumerStatefulWidget {
  final String categoryName;
  final List<Transaction> transactions;

  const CategoryBottomSheetContent({
    Key? key,
    required this.categoryName,
    required this.transactions,
  }) : super(key: key);

  @override
  ConsumerState<CategoryBottomSheetContent> createState() =>
      _CategoryBottomSheetContentState();
}

class _CategoryBottomSheetContentState
    extends ConsumerState<CategoryBottomSheetContent> {
  late List<Transaction> _transactions;

  @override
  void initState() {
    super.initState();
    _transactions = List.from(widget.transactions);
  }

  @override
  Widget build(BuildContext context) {
    // 숫자 포맷터
    final numberFormat = NumberFormat('#,###', 'ko_KR');

    // 총 지출 금액 계산
    int totalAmount = _transactions.fold<int>(
        0, (sum, transaction) => sum + transaction.householdAmount);

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // 카테고리 제목
        Text(widget.categoryName, style: AppTextStyles.modalTitle),

        // 총 건수 및 지출 정보
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
            Text(
              '${numberFormat.format(totalAmount)}원',
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
                  itemCount: _transactions.length,
                  itemBuilder: (context, index) {
                    final transaction = _transactions[index];
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: TransactionListItem(
                        transaction: transaction,
                      ),
                    );
                  },
                ),
        ),
      ],
    );
  }
}

// 바텀시트 모달을 표시하는 함수
void showCategoryBottomSheetModal({
  required BuildContext context,
  required WidgetRef ref,
  required String categoryName,
  required List<Transaction> transactions,
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
    child: CategoryBottomSheetContent(
      categoryName: categoryName,
      transactions: transactions,
    ),
  );
}
