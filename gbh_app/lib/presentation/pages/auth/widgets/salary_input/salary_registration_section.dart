import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/my/salary_model.dart';

class SalaryRegistrationSection extends StatelessWidget {
  final AccountModel account;
  final DepositModel deposit;
  final bool isLoading;
  final String? errorMessage;

  const SalaryRegistrationSection({
    Key? key,
    required this.account,
    required this.deposit,
    this.isLoading = false,
    this.errorMessage,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // 숫자 포맷터 (천 단위 콤마)
    final currencyFormatter = NumberFormat('#,###', 'ko_KR');
    
    // 날짜 포맷팅 (예: "20250328" -> "2025.03.28")
    final dateStr = deposit.transactionDate;
    final formattedDate = "${dateStr.substring(0, 4)}.${dateStr.substring(4, 6)}.${dateStr.substring(6, 8)}";
    
    // 월급일(일자만) 추출 (예: "20250328" -> "28일")
    final salaryDay = "${dateStr.substring(6, 8)}일";

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
              ],
            ),
          )
        // 월급 등록 정보 표시
        else
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: AppColors.background,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildInfoRow('월급 계좌', '${account.bankName} ${_formatAccountNumber(account.accountNo)}'),
                const SizedBox(height: 16),
                _buildInfoRow('월급일', salaryDay),
                const SizedBox(height: 16),
                _buildInfoRow('월급액', '${currencyFormatter.format(deposit.transactionBalance)}원'),
                const SizedBox(height: 16),
                _buildInfoRow('입금 내역', deposit.transactionSummary),
                if (deposit.transactionMemo.isNotEmpty) ...[
                  const SizedBox(height: 16),
                  _buildInfoRow('메모', deposit.transactionMemo),
                ],
                const SizedBox(height: 40),
                Text(
                  '* 월급 정보는 나중에 마이페이지에서 수정할 수 있습니다.',
                  style: AppTextStyles.bodySmall.copyWith(
                    color: AppColors.textSecondary,
                    fontStyle: FontStyle.italic,
                    fontWeight: FontWeight.w300
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 80,
          child: Text(
            label,
            style: AppTextStyles.bodyMedium.copyWith(
              color: AppColors.textSecondary,
            ),
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Text(
            value,
            style: AppTextStyles.bodyMedium,
          ),
        ),
      ],
    );
  }

  // 계좌번호 포맷팅 (예: 1234-5678-9012 -> 1234-****-9012)
  String _formatAccountNumber(String accountNo) {
    if (accountNo.length < 8) return accountNo;
   
    final firstPart = accountNo.substring(0, 4);
    final lastPart = accountNo.substring(accountNo.length - 4);
    
    return '$firstPart-****-$lastPart';
  }
}