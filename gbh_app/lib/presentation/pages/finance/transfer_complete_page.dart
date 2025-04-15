// lib/presentation/pages/finance/transfer_complete_page.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/utils/format_utils.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/finance/transfer_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';

class TransferCompletePage extends ConsumerWidget {
  final String withdrawalAccountNo;
  final String depositAccountNo;
  final int amount;

  const TransferCompletePage({
    Key? key,
    required this.withdrawalAccountNo,
    required this.depositAccountNo,
    required this.amount,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: CustomAppbar(
        title: 'my little 자산',
        backgroundColor: AppColors.background,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Lottie.asset(
              'assets/images/loading/success.json',
              width: 140,
              height: 140,
              fit: BoxFit.contain,
            ),
            const SizedBox(height: 20),
            Text('${NumberFormat.formatWithComma(amount.toString())}원',
                style: AppTextStyles.bodyLarge.copyWith(fontWeight: FontWeight.w600)),
            const Text(
              '송금이 완료되었습니다',
              style: AppTextStyles.bodyLarge
            ),
            const SizedBox(height: 30),
            
            // 송금 정보 요약
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(10),
                border: Border.all(width: 1.0, color: AppColors.blackLight)
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('출금 계좌', style: AppTextStyles.bodyMedium.copyWith(color: AppColors.disabled, fontWeight: FontWeight.w400)),
                      Text(formatAccountNumber(withdrawalAccountNo), style: AppTextStyles.bodyMedium),
                    ],
                  ),
                  const SizedBox(height: 8),
                  
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('입금 계좌', style: AppTextStyles.bodyMedium.copyWith(color: AppColors.disabled, fontWeight: FontWeight.w400)),
                      Text(formatAccountNumber(depositAccountNo), style: AppTextStyles.bodyMedium),
                    ],
                  ),
                  const SizedBox(height: 8),
                  
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('보낸 금액', style: AppTextStyles.bodyMedium.copyWith(color: AppColors.disabled, fontWeight: FontWeight.w400)),
                      Text('${NumberFormat.formatWithComma(amount.toString())}원', style: AppTextStyles.bodyMedium),
                    ],
                  ),
                ],
              ),
            ),
            
            const Spacer(),
            
            // 확인 버튼
            Button(
              onPressed: () {
                ref.read(transferProvider.notifier).reset();
                // 계좌 정보 새로고침
                final financeViewModel = ref.read(financeViewModelProvider.notifier);
                financeViewModel.refreshAssetInfo();
                
                // 메인 화면으로 이동
                context.go(FinanceRoutes.root);
              },
              text: '확인',
            ),
            const SizedBox(height: 10),
          ],
        ),
      ),
    );
  }
}

// 계좌번호 포맷팅 함수 (기존 코드에서 가져옴)
String formatAccountNumber(String accountNo) {
  String numbersOnly = accountNo.replaceAll(RegExp(r'[^0-9]'), '');
  String formatted = '';
  
  for (int i = 0; i < numbersOnly.length; i++) {
    if (i > 0 && i % 4 == 0) {
      formatted += '-';
    }
    formatted += numbersOnly[i];
  }
  
  return formatted;
}