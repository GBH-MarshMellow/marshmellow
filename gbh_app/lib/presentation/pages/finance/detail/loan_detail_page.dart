// lib/presentation/pages/finance/loan_detail_page.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/finance/detail/loan_detail_model.dart';
import 'package:marshmellow/presentation/viewmodels/finance/loan_detail_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/finance/bank_icon.dart';

class LoanDetailPage extends ConsumerWidget {
  final String accountNo;
  final String bankName;
  final String accountName;
  final int balance;

  const LoanDetailPage({
    Key? key,
    required this.accountNo,
    required this.bankName,
    required this.accountName,
    required this.balance,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final loanDetailsAsync = ref.watch(loanPaymentDetailsProvider(accountNo));

    return Scaffold(
      appBar: CustomAppbar(
        title: 'my little 자산',
      ),
      body: loanDetailsAsync.when(
        data: (response) => _buildContent(context, response.data),
        loading: () => Center(
          child: Lottie.asset(
            'assets/images/loading/loading_simple.json',
            width: 140,
            height: 140,
            fit: BoxFit.contain,
          ),
        ),
        error: (error, stack) => Center(
          child: Text('오류가 발생했습니다: $error', textAlign: TextAlign.center),
        ),
      ),
    );
  }

  Widget _buildContent(BuildContext context, LoanDetailData data) {
    return Column(
      children: [
        _buildLoanHeader(data),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Row(
            children: [
              Text(
                '상환 내역',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ],
          ),
        ),
        Expanded(
          child: _buildRepaymentList(data.repaymentRecords),
        ),
      ],
    );
  }

  Widget _buildLoanHeader(LoanDetailData data) {
    return Container(
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
          border: Border.all(width: 1.0, color: AppColors.divider),
          borderRadius: const BorderRadius.all(Radius.circular(10.0))),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(accountName, style: AppTextStyles.bodyLarge),
          const SizedBox(height: 4),
          Row(
            children: [
              BankIcon(bankName: bankName, size: 20),
              const SizedBox(width: 4),
              Text(
                accountNo,
                style:
                    AppTextStyles.bodyMedium.copyWith(color: AppColors.divider),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('대출 상태', style: AppTextStyles.bodyMedium),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: data.status == '연체'
                      ? AppColors.warnningLight.withOpacity(0.1)
                      : AppColors.blueDark.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  data.status,
                  style: AppTextStyles.bodySmall.copyWith(
                    fontWeight: FontWeight.bold,
                    color: data.status == '연체'
                        ? AppColors.warnningLight
                        : AppColors.blueDark,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('대출금', style: AppTextStyles.bodyMedium),
              Text(
                '${NumberFormat('#,###').format(int.tryParse(data.loanBalance) ?? 0)}원',
                style: AppTextStyles.bodyMedium,
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('남은 상환금액',
                  style: AppTextStyles.bodyMedium
                      .copyWith(color: AppColors.divider)),
              Text(
                // '${NumberFormat('#,###').format(data.remainingLoanBalance)}원',
                '${NumberFormat('#,###').format(int.tryParse(data.remainingLoanBalance) ?? 0)}원',
                style: AppTextStyles.bodyExtraLarge,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildRepaymentList(List<RepaymentRecord> records) {
    if (records.isEmpty) {
      return const Center(
        child: Text('상환 내역이 없습니다.'),
      );
    }

    // 납부 예정일 기준으로 그룹화
    final Map<String, List<RepaymentRecord>> groupedRecords = {};

    for (var record in records) {
      final date = record.repaymentAttemptDate.substring(0, 6); // 년월 기준으로 그룹화
      if (!groupedRecords.containsKey(date)) {
        groupedRecords[date] = [];
      }
      groupedRecords[date]!.add(record);
    }

    // 날짜 기준으로 정렬 (최신순)
    final sortedDates = groupedRecords.keys.toList()
      ..sort((a, b) => b.compareTo(a));

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: sortedDates.length,
      itemBuilder: (context, dateIndex) {
        final date = sortedDates[dateIndex];
        final dateRecords = groupedRecords[date]!;

        // 같은 월의 항목들을 납부 예정일순으로 정렬
        dateRecords.sort(
            (a, b) => b.repaymentAttemptDate.compareTo(a.repaymentAttemptDate));

        final month = date.substring(4, 6);
        final year = date.substring(0, 4);

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 월 헤더
            Padding(
              padding: const EdgeInsets.only(top: 16, bottom: 16),
              child: Text(
                '$year년 $month월',
                style: AppTextStyles.bodyMedium.copyWith(color: Colors.grey),
              ),
            ),

            // 해당 월의 상환 항목들
            ...dateRecords.map((record) {
              final isSuccess = record.status == 'SUCCESS';

              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 회차 및 납부 정보
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '${record.installmentNumber}회차',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w400,
                              color: AppColors.textPrimary,
                            ),
                            overflow: TextOverflow.ellipsis,
                          ),
                          Row(
                            children: [
                              Text(
                                '납부 예정일: ${_formatDayDate(record.repaymentAttemptDate)}',
                                style: AppTextStyles.bodySmall
                                    .copyWith(color: Colors.grey),
                              ),
                            ],
                          ),
                          if (isSuccess && record.repaymentActualDate != null)
                            Text(
                              '납부일: ${_formatDayDate(record.repaymentActualDate!)}',
                              style: AppTextStyles.bodySmall
                                  .copyWith(color: Colors.grey),
                            ),
                          if (!isSuccess && record.failureReason.isNotEmpty)
                            Text(
                              '실패 사유: ${record.failureReason}',
                              style: AppTextStyles.bodySmall
                                  .copyWith(color: AppColors.warnningLight),
                            ),
                        ],
                      ),
                    ),

                    // 금액 및 상태
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          '${NumberFormat('#,###').format(int.parse(record.paymentBalance))}원',
                          style: AppTextStyles.bodyMedium.copyWith(
                            fontWeight: FontWeight.bold,
                            color: isSuccess
                                ? AppColors.blueDark
                                : AppColors.warnningLight,
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: isSuccess
                                ? AppColors.blueDark.withOpacity(0.1)
                                : AppColors.warnningLight.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            isSuccess ? '상환 완료' : '상환 실패',
                            style: AppTextStyles.bodySmall.copyWith(
                              color: isSuccess
                                  ? AppColors.blueDark
                                  : AppColors.warnningLight,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              );
            }).toList(),

            // 월 그룹 사이 여백
            SizedBox(height: dateIndex < sortedDates.length - 1 ? 8 : 0),
          ],
        );
      },
    );
  }

  String _formatDayDate(String dateStr) {
    final month = dateStr.substring(4, 6);
    final day = dateStr.substring(6, 8);
    return '$month.$day';
  }
}
