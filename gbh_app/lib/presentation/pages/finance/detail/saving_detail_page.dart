// lib/presentation/pages/finance/saving_detail_page.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/finance/detail/saving_detail_model.dart';
import 'package:marshmellow/presentation/viewmodels/finance/saving_detail_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/finance/bank_icon.dart';

class SavingDetailPage extends ConsumerWidget {
  final String accountNo;
  final String bankName;
  final String accountName;
  final int balance;
  final bool noMoneyMan;

  const SavingDetailPage({
    Key? key,
    required this.accountNo,
    required this.bankName,
    required this.accountName,
    required this.balance,
    required this.noMoneyMan,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // 계좌번호로 적금 납입내역 조회
    final paymentsAsync = ref.watch(savingPaymentsProvider(accountNo));

    return Scaffold(
      appBar: CustomAppbar(
        title: 'my little 자산',
      ),
      body: paymentsAsync.when(
        data: (response) {
          // 데이터가 있는 경우
          if (response.data.paymentList.paymentInfo.isEmpty) {
            return const Center(
              child: Text('납입 내역이 없습니다.'),
            );
          }

          // 첫 번째 적금 정보 가져오기 (API 응답에서 하나의 적금만 반환된다고 가정)
          final savingItem = response.data.paymentList;

          return Column(
            children: [
              _buildSavingHeader(savingItem),
              _buildSavingInfo(savingItem),
              Expanded(
                child: _buildPaymentList(savingItem.paymentInfo),
              ),
            ],
          );
        },
        loading: () => Center(
          child: Lottie.asset(
            'assets/images/loading/loading_simple.json',
            width: 140,
            height: 140,
            fit: BoxFit.contain,
          ),
        ),
        error: (error, stack) => Center(
          child: Text('오류가 발생했습니다: $error'),
        ),
      ),
    );
  }

  // 적금 계좌 정보 헤더
  Widget _buildSavingHeader(SavingPaymentItem item) {
    final formatter = NumberFormat('#,###');

    return Container(
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
          border: Border.all(width: 1.0, color: AppColors.divider),
          borderRadius: const BorderRadius.all(Radius.circular(10.0))),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              BankIcon(bankName: bankName, size: 46),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(accountName, style: AppTextStyles.bodyLarge),
                  Text('${bankName} | ${_formatAccountNumber(accountNo)}',
                      style: AppTextStyles.bodyMedium
                          .copyWith(color: AppColors.divider)),
                ],
              ),
            ],
          ),
          const SizedBox(height: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('적금 총액',
                  style: AppTextStyles.bodyMedium
                      .copyWith(color: AppColors.divider)),
              Text('${formatter.format(int.parse(item.totalBalance))}원',
                  style: AppTextStyles.bodyExtraLarge),
            ],
          ),
        ],
      ),
    );
  }

  // 카드번호 형식화
  String _formatAccountNumber(String accountNo) {
    if (accountNo.length <= 4) return accountNo;
    return '${accountNo.substring(0, 4)}...${accountNo.substring(accountNo.length - 4)}';
  }

  // 적금 상세 정보
  Widget _buildSavingInfo(SavingPaymentItem item) {
    final createDate = _formatDateForDisplay(item.accountCreateDate);
    final expiryDate = _formatDateForDisplay(item.accountExpiryDate);
    final formatter = NumberFormat('#,###');

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      margin: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
          color: AppColors.background,
          border: Border.all(width: 1.0, color: AppColors.divider),
          borderRadius: const BorderRadius.all(Radius.circular(10.0))),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('적금 정보',
              style: AppTextStyles.bodyMedium
                  .copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          _buildInfoRow(
              '월 납입액', '${formatter.format(int.parse(item.depositBalance))}원'),
          _buildInfoRow('적금 금리', '연 ${item.interestRate}%'),
          _buildInfoRow('가입일', createDate),
          _buildInfoRow('만기일', expiryDate),
        ],
      ),
    );
  }

  // 정보 행 위젯
  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: AppTextStyles.bodyMedium
                  .copyWith(color: AppColors.textSecondary)),
          Text(value, style: AppTextStyles.bodyMedium),
        ],
      ),
    );
  }

  // 납입 내역 리스트
  Widget _buildPaymentList(List<PaymentInfoItem> payments) {
    if (payments.isEmpty) {
      return const Center(
        child: Text('납입 내역이 없습니다.'),
      );
    }

    // 날짜별로 거래 내역 그룹화
    final Map<String, List<PaymentInfoItem>> groupedPayments = {};

    for (var item in payments) {
      final date = item.paymentDate;
      if (!groupedPayments.containsKey(date)) {
        groupedPayments[date] = [];
      }
      groupedPayments[date]!.add(item);
    }

    // 날짜 기준으로 정렬 (최신순)
    final sortedDates = groupedPayments.keys.toList()
      ..sort((a, b) => b.compareTo(a));

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: sortedDates.length,
      itemBuilder: (context, dateIndex) {
        final date = sortedDates[dateIndex];
        final dateItems = groupedPayments[date]!;

        // 같은 날짜의 항목들을 시간순으로 정렬
        dateItems.sort((a, b) => b.paymentTime.compareTo(a.paymentTime));

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 날짜 헤더
            Padding(
              padding: const EdgeInsets.only(top: 16),
              child: Text(
                _formatTransactionDate(date),
                style: AppTextStyles.bodyMedium.copyWith(color: Colors.grey),
              ),
            ),
            Divider(
              color: AppColors.disabled,
            ),
            SizedBox(
              height: 16,
            ),
            // 해당 날짜의 납입 항목들
            ...dateItems.map((item) {
              final isSuccess = item.status == 'SUCCESS';
              final formatter = NumberFormat('#,###');

              return Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 납입 내용
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '${item.depositInstallment}회차 납입',
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
                                _formatTimeForDisplay(item.paymentTime),
                                style: AppTextStyles.bodySmall
                                    .copyWith(color: Colors.grey),
                              ),
                              Text(' | ',
                                  style: AppTextStyles.bodySmall
                                      .copyWith(color: Colors.grey)),
                              Text(
                                isSuccess ? '납입 완료' : '납입 실패',
                                style: AppTextStyles.bodySmall.copyWith(
                                    color: isSuccess
                                        ? Colors.grey
                                        : AppColors.warnningLight),
                              ),
                            ],
                          ),
                          if (item.failureReason != null &&
                              item.failureReason!.isNotEmpty)
                            Padding(
                              padding: const EdgeInsets.only(top: 4),
                              child: Text(
                                '실패 사유: ${item.failureReason}',
                                style: AppTextStyles.bodySmall.copyWith(
                                  color: AppColors.warnningLight,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),

                    // 금액 및 상태
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          '${formatter.format(int.parse(item.paymentBalance))}원',
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
                            isSuccess ? '성공' : '실패',
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

            // 날짜 그룹 사이 여백
            SizedBox(height: dateIndex < sortedDates.length - 1 ? 8 : 0),
          ],
        );
      },
    );
  }

  // 날짜 포맷 변환 (20250321 -> 2025.03.21)
  String _formatDateForDisplay(String dateStr) {
    if (dateStr.length != 8) return dateStr;

    final year = dateStr.substring(0, 4);
    final month = dateStr.substring(4, 6);
    final day = dateStr.substring(6, 8);
    return '$year.$month.$day';
  }

  // 날짜 포맷 변환 (요일 포함)
  String _formatTransactionDate(String dateStr) {
    final year = int.parse(dateStr.substring(0, 4));
    final month = int.parse(dateStr.substring(4, 6));
    final day = int.parse(dateStr.substring(6, 8));

    final date = DateTime(year, month, day);
    final weekdayName = _getWeekdayName(date.weekday);

    return '$month월 $day일 $weekdayName';
  }

  String _getWeekdayName(int weekday) {
    switch (weekday) {
      case 1:
        return '월요일';
      case 2:
        return '화요일';
      case 3:
        return '수요일';
      case 4:
        return '목요일';
      case 5:
        return '금요일';
      case 6:
        return '토요일';
      case 7:
        return '일요일';
      default:
        return '';
    }
  }

  // 시간 포맷 변환 (151915 -> 15:19)
  String _formatTimeForDisplay(String timeStr) {
    if (timeStr.length < 4) return timeStr;

    final hour = timeStr.substring(0, 2);
    final minute = timeStr.substring(2, 4);
    return '$hour:$minute';
  }
}
