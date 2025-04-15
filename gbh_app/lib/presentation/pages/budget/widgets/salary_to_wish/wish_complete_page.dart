import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/wishlist/wish_model.dart';
import 'package:marshmellow/data/models/wishlist/wishlist_model.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/celebration/celebration.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/finance/bank_icon.dart';

class WishCompletePage extends ConsumerWidget {
  final Wishlist wishlist;
  final int selectedMonth;
  final int dailyAmount;
  final DemDepItem withdrawalAccount;
  final DemDepItem depositAccount;
  final String dueDate;

  const WishCompletePage({
    Key? key,
    required this.wishlist,
    required this.selectedMonth,
    required this.dailyAmount,
    required this.withdrawalAccount,
    required this.depositAccount,
    required this.dueDate,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // 종료일 포맷팅
    final year = dueDate.substring(0, 4);
    final month = dueDate.substring(4, 6);
    final day = dueDate.substring(6, 8);
    final formattedDueDate = '$year년 $month월 $day일';
    
    return Scaffold(
      appBar: CustomAppbar(
        title: '위시 등록 완료',
        automaticallyImplyLeading: false
      ),
      body: Stack(
        children: [
          // 메인 콘텐츠
          SingleChildScrollView(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                const SizedBox(height: 40),
                Icon(
                  Icons.check_circle,
                  size: 80,
                  color: AppColors.blueDark,
                ),
                const SizedBox(height: 24),
                Text(
                  '위시 등록과 자동이체 설정이 완료되었습니다!',
                  style: AppTextStyles.bodyMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 12),
                Text(
                  '$selectedMonth개월 동안 매일 저축을 통해\n위시 상품을 모아보세요!',
                  style: AppTextStyles.bodyMedium.copyWith(
                    color: AppColors.textSecondary,
                    fontWeight: FontWeight.w300
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 40),
                
                // 위시 정보 카드
                _buildInfoCard(
                  title: '위시 정보',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildInfoRow('상품명', wishlist.productName),
                      _buildInfoRow('금액', '${_formatNumber(wishlist.productPrice)}원'),
                      _buildInfoRow('저축 기간', '$selectedMonth개월'),
                      _buildInfoRow('일 저축 금액', '${_formatNumber(dailyAmount)}원'),
                      _buildInfoRow('종료일', formattedDueDate),
                    ],
                  ),
                ),
                
                const SizedBox(height: 20),
                
                // 계좌 정보 카드
                _buildInfoCard(
                  title: '자동이체 정보',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildAccountRow('출금계좌', withdrawalAccount),
                      _buildAccountRow('입금계좌', depositAccount),
                      _buildInfoRow('이체 금액', '${_formatNumber(dailyAmount)}원 (매일)'),
                    ],
                  ),
                ),
                
                const SizedBox(height: 40),
                
                // 메인 버튼
                ElevatedButton(
                  onPressed: () {
                    // 메인 페이지로 이동
                    context.go('/budget');
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.backgroundBlack,
                    minimumSize: const Size(double.infinity, 50),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  child: Text(
                    '메인으로',
                    style: AppTextStyles.bodyLarge.copyWith(
                      color: Colors.white,
                      fontWeight: FontWeight.w300
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoCard({required String title, required Widget child}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
                  color:AppColors.backgroundBlack,  // 원하는 테두리 색상
                  width: 0.5,                   // 테두리 두께
                )
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: AppTextStyles.bodyLarge.copyWith(
              fontWeight: FontWeight.w300,
            ),
          ),
          const Divider(height: 10),
          child,
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 150,
            child: Text(
              label,
              style: AppTextStyles.bodyMedium.copyWith(
                color: AppColors.textSecondary,
                fontWeight: FontWeight.w300
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: AppTextStyles.bodyMedium,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAccountRow(String label, DemDepItem account) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              label,
              style: AppTextStyles.bodyMedium.copyWith(
                color: AppColors.textSecondary,
              ),
            ),
          ),
          BankIcon(
            bankName: account.bankName,
            size: 24,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              '${account.bankName} ${_formatAccountNumber(account.accountNo)}',
              style: AppTextStyles.bodyMedium,
            ),
          ),
        ],
      ),
    );
  }

  String _formatNumber(int number) {
    return number.toString().replaceAllMapped(
          RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
          (Match m) => '${m[1]},',
        );
  }

  String _formatAccountNumber(String accountNo) {
    // 계좌번호 포맷 (예: 123-456-789)
    if (accountNo.length < 6) return accountNo;
    
    final midPos = (accountNo.length / 2).round();
    final first = accountNo.substring(0, midPos);
    final last = accountNo.substring(midPos);
    
    return '$first-$last';
  }
}