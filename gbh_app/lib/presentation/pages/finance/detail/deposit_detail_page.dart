// lib/presentation/pages/finance/deposit_detail_page.dart

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:intl/intl.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/finance/detail/deposit_detail_model.dart';
import 'package:marshmellow/presentation/viewmodels/finance/deposit_detail_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/finance/bank_icon.dart';

class DepositDetailPage extends ConsumerWidget {
  final String accountNo;
  final String bankName;
  final String accountName;
  final int balance;
  final bool noMoneyMan;

  const DepositDetailPage({
    Key? key,
    required this.accountNo,
    required this.bankName,
    required this.accountName,
    required this.balance,
    required this.noMoneyMan,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final depositPaymentAsync = ref.watch(depositPaymentProvider(accountNo));

    return Scaffold(
      appBar: CustomAppbar(
        title: 'my little 자산',
      ),
      body: Column(
        children: [
          _buildAccountHeader(context),
          Expanded(
            child: depositPaymentAsync.when(
              data: (response) =>
                  _buildPaymentDetails(context, response.data.payment),
              loading: () => Center(
                child: Lottie.asset(
                  'assets/images/loading/loading_simple.json',
                  width: 140,
                  height: 140,
                  fit: BoxFit.contain,
                ),
              ),
              error: (error, stack) =>
                  Center(child: Text('오류가 발생했습니다: $error')),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAccountHeader(BuildContext context) {
    return Container(
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
          border: Border.all(width: 1.0, color: AppColors.divider),
          borderRadius: BorderRadius.circular(10.0)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(accountName, style: AppTextStyles.bodyLarge),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // 은행아이콘
              BankIcon(bankName: bankName, size: 24),
              const SizedBox(width: 8),
              // 은행명
              Text(
                '$bankName | ',
                style:
                    AppTextStyles.bodyMedium.copyWith(color: AppColors.divider),
              ),
              //계좌번호
              Text(
                _formatAccountNumber(accountNo),
                style:
                    AppTextStyles.bodyMedium.copyWith(color: AppColors.divider),
              ),
              const SizedBox(width: 4),
              //복사버튼
              InkWell(
                onTap: () {
                  Clipboard.setData(ClipboardData(text: accountNo));
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('계좌번호가 복사되었습니다')),
                  );
                },
                borderRadius: BorderRadius.circular(4),
                child: Padding(
                  padding: const EdgeInsets.all(4),
                  child: SvgPicture.asset(
                    'assets/icons/body/CopySimple.svg',
                    height: 16,
                    color: AppColors.disabled,
                  ),
                ),
              )
            ],
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('총 예금액',
                      style: AppTextStyles.bodyMedium
                          .copyWith(color: AppColors.divider)),
                  Text(
                    '${NumberFormat('#,###').format(balance)}원',
                    style: AppTextStyles.bodyExtraLarge,
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildPaymentDetails(BuildContext context, PaymentItem payment) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '예금 상세 정보',
            style:
                AppTextStyles.bodyLarge.copyWith(fontWeight: FontWeight.w400),
          ),
          const SizedBox(height: 16),
          _buildInfoCard('예금 정보', [
            _buildDetailRow('예금 종류', accountName),
            _buildDetailRow('예금 계좌', _formatAccountNumber(accountNo)),
            _buildDetailRow('가입일', _getRelativeTimeString(payment.paymentDate)),
            _buildDetailRow('만기일', _getExpiryDate(payment.paymentDate)),
          ]),
          const SizedBox(height: 16),
          _buildInfoCard('납입 정보', [
            _buildDetailRow('최근 납입액',
                '${NumberFormat('#,###').format(int.parse(payment.paymentBalance))}원'),
            // _buildDetailRow('납입 횟수', '${payment.paymentUniqueNo}회'),
            _buildDetailRow('납입 일시',
                _formatDateTime(payment.paymentDate, payment.paymentTime)),
            _buildDetailRow('연 이자율', '3.5%'), // 예시 데이터
          ]),
          const SizedBox(height: 16),
          _buildInfoCard('만기 예상', [
            _buildDetailRow(
                '만기 예상액', '${NumberFormat('#,###').format(balance * 1.08)}원'),
            _buildDetailRow(
                '예상 이자', '${NumberFormat('#,###').format(balance * 0.08)}원'),
            _buildDetailRow('세전 수익률', '8.0%'),
          ]),
        ],
      ),
    );
  }

  Widget _buildInfoCard(String title, List<Widget> rows) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.divider.withOpacity(0.3)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: AppTextStyles.bodyMedium.copyWith(
              fontWeight: FontWeight.bold,
              color: AppColors.blueDark,
            ),
          ),
          const SizedBox(height: 12),
          ...rows.map((row) {
            final index = rows.indexOf(row);
            return Column(
              children: [
                row,
                if (index < rows.length - 1) const SizedBox(height: 12),
              ],
            );
          }).toList(),
        ],
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style:
              AppTextStyles.bodyMedium.copyWith(color: AppColors.textSecondary),
        ),
        Text(
          value,
          style: AppTextStyles.bodyMedium,
        ),
      ],
    );
  }

  String _formatAccountNumber(String accountNo) {
    // 계좌번호 형식화 (은행마다 다를 수 있음)
    if (accountNo.length < 10) return accountNo;

    // 마지막 4자리만 표시
    return '***-${accountNo.substring(accountNo.length - 4, accountNo.length)}';
  }

  String _formatDateTime(String date, String time) {
    // 날짜 형식: YYYYMMDD, 시간 형식: HHMMSS
    final year = date.substring(0, 4);
    final month = date.substring(4, 6);
    final day = date.substring(6, 8);

    final hour = time.substring(0, 2);
    final minute = time.substring(2, 4);

    return '$year.$month.$day $hour:$minute';
  }

  String _getRelativeTimeString(String dateStr) {
    // 예시: 날짜를 기준으로 "N개월 전" 형식으로 변환
    final year = int.parse(dateStr.substring(0, 4));
    final month = int.parse(dateStr.substring(4, 6));
    final day = int.parse(dateStr.substring(6, 8));

    final date = DateTime(year, month, day);
    final now = DateTime.now();
    final difference = now.difference(date);

    final months = ((now.year - date.year) * 12) + now.month - date.month;

    if (months > 0) {
      return '$months개월 전';
    } else if (difference.inDays > 0) {
      return '${difference.inDays}일 전';
    } else {
      return '오늘';
    }
  }

  String _getExpiryDate(String startDateStr) {
    // 가입일로부터 1년 후의 날짜 계산 (예시)
    final year = int.parse(startDateStr.substring(0, 4));
    final month = int.parse(startDateStr.substring(4, 6));
    final day = int.parse(startDateStr.substring(6, 8));

    final startDate = DateTime(year, month, day);
    final expiryDate =
        DateTime(startDate.year + 1, startDate.month, startDate.day);

    return DateFormat('yyyy.MM.dd').format(expiryDate);
  }
}
