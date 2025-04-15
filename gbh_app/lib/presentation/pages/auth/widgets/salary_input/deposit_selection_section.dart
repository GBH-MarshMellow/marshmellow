import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/my/salary_model.dart';

class DepositSelectionSection extends StatelessWidget {
  final List<DepositModel> deposits;
  final DepositModel? selectedDeposit;
  final Function(DepositModel) onDepositSelected;
  final bool isLoading;
  final String? errorMessage;

  const DepositSelectionSection({
    Key? key,
    required this.deposits,
    required this.selectedDeposit,
    required this.onDepositSelected,
    this.isLoading = false,
    this.errorMessage,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // 로딩 표시
        if (isLoading)
          const Center(
            child: CircularProgressIndicator(),
          )
        // 오류 메시지 표시
        else if (errorMessage != null)
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(
                  Icons.error_outline,
                  color: Colors.red,
                  size: 48,
                ),
                const SizedBox(height: 16),
                Text(
                  errorMessage!,
                  style: AppTextStyles.bodyMedium.copyWith(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: () {
                    // 재시도 로직
                  },
                  child: const Text('다시 시도'),
                ),
              ],
            ),
          )
        // 입금 내역이 비어있을 때
        else if (deposits.isEmpty)
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(
                  Icons.monetization_on_outlined,
                  color: AppColors.textSecondary,
                  size: 48,
                ),
                const SizedBox(height: 16),
                Text(
                  '입금 내역이 없습니다.',
                  style: AppTextStyles.bodyMedium.copyWith(color: AppColors.textSecondary),
                ),
              ],
            ),
          )
        // 입금 내역 목록 표시
        else
          ListView.separated(
            physics: const NeverScrollableScrollPhysics(),
            shrinkWrap: true,
            itemCount: deposits.length,
            separatorBuilder: (context, index) => const Divider(height: 1),
            itemBuilder: (context, index) {
              final deposit = deposits[index];
              final isSelected = selectedDeposit != null && 
                                selectedDeposit?.transactionDate == deposit.transactionDate &&
                                selectedDeposit?.transactionTime == deposit.transactionTime &&
                                selectedDeposit?.transactionBalance == deposit.transactionBalance;

              return _buildDepositItem(deposit, isSelected);
            },
          ),
      ],
    );
  }

  Widget _buildDepositItem(DepositModel deposit, bool isSelected) {
    // 숫자 포맷터 (천 단위 콤마)
    final currencyFormatter = NumberFormat('#,###', 'ko_KR');
    
    // 날짜 포맷팅 (예: "20250328" -> "2025.03.28")
    final dateStr = deposit.transactionDate;
    final formattedDate = "${dateStr.substring(0, 4)}.${dateStr.substring(4, 6)}.${dateStr.substring(6, 8)}";
    
    // 시간 포맷팅 (예: "101452" -> "10:14:52")
    final timeStr = deposit.transactionTime;
    final formattedTime = "${timeStr.substring(0, 2)}:${timeStr.substring(2, 4)}:${timeStr.substring(4, 6)}";

    return InkWell(
      onTap: () => onDepositSelected(deposit),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 16.0, horizontal: 8.0),
        child: Row(
          children: [
            // 날짜 및 금액 정보
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(
                        formattedDate,
                        style: AppTextStyles.bodySmall.copyWith(
                          color: AppColors.textSecondary,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        formattedTime,
                        style: AppTextStyles.bodySmall.copyWith(
                          color: AppColors.textSecondary,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${currencyFormatter.format(deposit.transactionBalance)}원',
                    style: AppTextStyles.bodyLargeLight,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    deposit.transactionSummary,
                    style: AppTextStyles.bodySmall,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  if (deposit.transactionMemo.isNotEmpty) ...[
                    const SizedBox(height: 2),
                    Text(
                      deposit.transactionMemo,
                      style: AppTextStyles.bodySmall.copyWith(
                        color: AppColors.textSecondary,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ],
              ),
            ),
            
            // 선택 표시
            if (isSelected)
              const Icon(
                Icons.check_circle,
                color: AppColors.backgroundBlack,
                size: 24,
              )
            else
              const Icon(
                Icons.circle_outlined,
                color: AppColors.textSecondary,
                size: 24,
              ),
          ],
        ),
      ),
    );
  }
}