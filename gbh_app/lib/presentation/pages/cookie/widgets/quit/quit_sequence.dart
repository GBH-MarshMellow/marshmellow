import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/presentation/viewmodels/quit/quit_viewmodel.dart';

class QuitSequence extends ConsumerWidget {
  const QuitSequence({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final quitState = ref.watch(quitViewModelProvider);
    final availableAmount = quitState.availableAmount;
    final averageSpending = quitState.averageSpending;

    if (quitState.errorMessage != null) {
      return Center(child: Text(quitState.errorMessage!));
    }

    if (availableAmount == null || averageSpending == null) {
      return const Center(child: Text('데이터를 불러올 수 없습니다.'));
    }

    // 초기 잔액 (사용 가능 금액)
    int balance = availableAmount.availableAmount;

    // 월별 잔액 계산 (최대 6개월)
    List<MonthlyBalance> monthlyBalances = [];

    // 현재 날짜 기준으로 다음 달부터 계산
    DateTime now = DateTime.now();
    DateTime startMonth = DateTime(now.year, now.month + 1, 1);

    for (int i = 0; i < 6; i++) {
      // 월별 잔액 계산
      balance -= averageSpending.averageMonthlySpending;

      // 잔액이 음수가 되면 더 이상 계산하지 않음
      if (balance < 0 && i > 0) break;

      // 월 이름과 잔액 저장
      String monthName = DateFormat('M월')
          .format(DateTime(startMonth.year, startMonth.month + i, 1));

      monthlyBalances.add(MonthlyBalance(monthName, balance));
    }

    return SizedBox(
      height: 420, // 카드 높이에 맞게 조정
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: monthlyBalances.length,
        itemBuilder: (context, index) {
          return _buildCard(
            context,
            index + 1, // 카드 번호 (1부터 시작)
            monthlyBalances[index].month,
            monthlyBalances[index].balance,
          );
        },
      ),
    );
  }

  Widget _buildCard(
      BuildContext context, int cardNumber, String month, int balance) {
    // 잔액 포맷팅
    final formatter = NumberFormat('#,###', 'ko_KR');
    final formattedBalance = formatter.format(balance);

    // 카드 이미지 경로 선택 (1~5까지는 그대로, 6은 음수일 때, 7은 양수일 때)
    String cardImagePath;
    if (cardNumber <= 5) {
      switch (cardNumber) {
        case 1:
          cardImagePath = IconPath.card1;
          break;
        case 2:
          cardImagePath = IconPath.card2;
          break;
        case 3:
          cardImagePath = IconPath.card3;
          break;
        case 4:
          cardImagePath = IconPath.card4;
          break;
        case 5:
          cardImagePath = IconPath.card5;
          break;
        default:
          cardImagePath = IconPath.card5; // 기본값
          break;
      }
    } else {
      // 6번째 카드부터는 잔액에 따라 다른 이미지 사용
      cardImagePath = balance >= 0 ? IconPath.card7 : IconPath.card6;
    }

    // 카드 6, 7번은 어두운 배경이므로 흰색 텍스트 사용 (이미지 경로로 체크)
    final Color textColor =
        (cardImagePath == IconPath.card6 || cardImagePath == IconPath.card7)
            ? Colors.white
            : AppColors.textPrimary;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 15.0),
      child: Stack(
        children: [
          // 카드 이미지
          Image.asset(
            cardImagePath,
            width: 250,
            height: 400,
            fit: BoxFit.contain,
          ),

          // 오버레이 텍스트
          Positioned(
            top: 50, // 상단 여백 조정
            left: 0,
            right: 0,
            child: Column(
              children: [
                Text(
                  month,
                  style: AppTextStyles.bodyLarge.copyWith(
                    color: textColor,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  '$formattedBalance원',
                  style: AppTextStyles.mainTitle.copyWith(
                    color: textColor,
                    fontWeight: FontWeight.bold,
                    fontSize: 24,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// 월별 잔액 데이터 클래스
class MonthlyBalance {
  final String month;
  final int balance;

  MonthlyBalance(this.month, this.balance);
}
