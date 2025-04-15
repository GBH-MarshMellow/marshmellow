import 'package:flutter/material.dart';
import 'package:flutter/src/widgets/framework.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/viewmodels/budget/budget_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';

/*
  예산 카테고리 리스트 페이지
  : 버블 차트의 배경을 탭하면 랜딩
*/
class BudgetCategoryDetailPage extends ConsumerStatefulWidget {
  final int budgetPk;

  const BudgetCategoryDetailPage({
    super.key,
    required this.budgetPk
  });

  @override
  ConsumerState<BudgetCategoryDetailPage> createState() => _BudgetCategoryDetailPageState();
}

  class _BudgetCategoryDetailPageState extends ConsumerState<BudgetCategoryDetailPage> {
    @override
    void initState() {
      super.initState();
      // 페이지 로드 될때 데이터 로드
      Future.microtask(() {
        ref.read(budgetProvider.notifier).fetchBudgets();
      });
    }

  @override
  Widget build(BuildContext context) {
    final budgetState = ref.watch(budgetProvider);

    // 선택된 예산 찾기
    final selectedBudget = budgetState.budgets.firstWhere(
      (budget) => budget.budgetPk == widget.budgetPk,
      orElse: () => budgetState.selectedBudget!,
    );

    if (budgetState.isLoading) {
      return Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    if (budgetState.errorMessage != null) {
      return Scaffold(
        appBar: const CustomAppbar(title: '예산 상세'),
        body: Center(
          child: Text('오류가 발생했습니다: ${budgetState.errorMessage}'),
        ),
      );
    }

    final categories = selectedBudget.budgetCategoryList;

    // 총예산/지출 계산
    final totalBudget = selectedBudget.budgetAmount;
    final totalSpent = categories.fold<int>(
      0,
      (sum, category) => sum + (category.budgetExpendAmount ?? 0),
    );

    // 남은예산 계산
    final remainingBudget = totalBudget - totalSpent;

    // 포맷팅
    final totalBudgetFormatted = _formatAmount(totalBudget);
    final totalSpentFormatted = _formatAmount(totalSpent);
    final remainingBudgetFormatted = _formatAmount(remainingBudget);

    return Scaffold(
      appBar: const CustomAppbar(title: '예산 상세'),
      body: Column(
        children: [
          const SizedBox(height: 6),
          // 예산 종합
          // 카테고리 목록 타이틀
          Padding(
            padding: const EdgeInsets.only(left: 16, right: 16, top: 8, bottom: 8),
            child: Row(
              children: [
                Text(
                  '이번 달 예산',
                  style: AppTextStyles.appBar.copyWith(
                    color: AppColors.textPrimary,
                    fontWeight: FontWeight.w300,
                  ),
                ),
              ],
            ),
          ),
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: AppColors.backgroundBlack, width: 0.5),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '총 예산',
                  style: AppTextStyles.bodyExtraSmall.copyWith(
                    color: AppColors.textPrimary,
                    fontWeight: FontWeight.w300,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  '$totalBudgetFormatted원',
                  style: AppTextStyles.mainTitle,
                ),
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    _buildSummaryItem(
                      '사용금액', 
                      '$totalSpentFormatted원',
                      totalSpent > totalBudget ? AppColors.buttonDelete : AppColors.pinkPrimary),
                    _buildSummaryItem('남은금액', '$remainingBudgetFormatted원', AppColors.bluePrimary),
                  ],
                ),
              ],
            ),
          ),

          const SizedBox(height: 25),

          // 카테고리 목록 타이틀
          Padding(
            padding: const EdgeInsets.only(left: 16, right: 16, top: 8, bottom: 8),
            child: Row(
              children: [
                Text(
                  '카테고리별 예산',
                  style: AppTextStyles.appBar.copyWith(
                    color: AppColors.textPrimary,
                    fontWeight: FontWeight.w300,
                  ),
                ),
              ],
            ),
          ),

          // 카테고리 목록
          Expanded(
            child: ListView.builder(
              itemCount: categories.length,
              itemBuilder: (context, index) {
                // Use a staggered layout to create the card stack effect
                final category = categories[index];
                final categoryColor = category.color;
                
                // 지출량 계산
                final percentage = (category.budgetExpendPercent ?? 0) * 100;
                final formattedBudget = _formatAmount(category.budgetCategoryPrice);
                final formattedSpent = _formatAmount(category.budgetExpendAmount ?? 0);
                
                return GestureDetector(
                  onTap: () {
                    // 카테고리별 지출 상세로 이동
                    context.push(
                      '/budget/category/expenses/${category.budgetCategoryPk}',
                      extra: {
                        'category': category,
                        'budgetPk': widget.budgetPk,
                      },
                    );
                  },
                  child: Container(
                    margin: EdgeInsets.only(
                      top: 4,
                      bottom: 4,
                      left: 16, // Staggered effect
                      right: 16,
                    ),
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: AppColors.backgroundBlack, width: 0.5),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Category Name & Color Indicator
                        Row(
                          children: [
                            Container(
                              width: 12,
                              height: 12,
                              decoration: BoxDecoration(
                                color: categoryColor,
                                shape: BoxShape.circle,
                              ),
                            ),
                            const SizedBox(width: 8),
                            Text(
                              category.budgetCategoryName,
                              style: AppTextStyles.bodyMedium.copyWith(
                                color: AppColors.textPrimary,
                                fontWeight: FontWeight.w300,
                              ),
                            ),
                            const Spacer(),
                            Text(
                              '${percentage.toStringAsFixed(0)}%',
                              style: AppTextStyles.bodySmall.copyWith(
                                color: percentage > 100 
                                    ? AppColors.buttonDelete 
                                    : AppColors.textSecondary,
                                fontWeight: percentage > 100 
                                    ? FontWeight.bold 
                                    : FontWeight.normal,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        
                        // 예산 상태바
                        ClipRRect(
                          borderRadius: BorderRadius.circular(10),
                          child: LinearProgressIndicator(
                            value: percentage / 100,
                            backgroundColor: AppColors.whiteDark,
                            valueColor: AlwaysStoppedAnimation<Color>(
                              percentage > 100 ? AppColors.buttonDelete : categoryColor,
                            ),
                            minHeight: 8,
                          ),
                        ),
                        const SizedBox(height: 12),
                        
                        // 예산 정보
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              '$formattedSpent원',
                              style: AppTextStyles.bodyMedium,
                            ),
                            Text(
                              '$formattedBudget원',
                              style: AppTextStyles.bodySmall.copyWith(
                                color: AppColors.textSecondary,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryItem(String label, String value, Color valueColor) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: AppTextStyles.bodyExtraSmall.copyWith(
            color: AppColors.textPrimary,
            fontWeight: FontWeight.w300,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: AppTextStyles.bodyMedium.copyWith(
            color: valueColor,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  String _formatAmount(int amount) {
    return amount.toString().replaceAllMapped(
      RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
      (Match m) => '${m[1]},'
    );
  }

}