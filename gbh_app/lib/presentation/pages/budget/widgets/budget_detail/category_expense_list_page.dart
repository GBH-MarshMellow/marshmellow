import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/budget/budget_model.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_detail_modal.dart';
import 'package:marshmellow/presentation/viewmodels/budget/budget_viewmodel.dart';
import 'package:marshmellow/data/repositories/budget/category_repository.dart';

/*
  예산 카테고리별 지출 프로바이더
*/
final categoryTransactionsProvider = FutureProvider.autoDispose
    .family<List<Transaction>, CategoryExpensePageParams>(
  (ref, params) async {
    final repository = ref.watch(categoryTransactionRepositoryProvider);

    // print('Provider 호출: budgetPk=${params.budgetPk}, categoryPk=${params.categoryPk}, categoryName=${params.categoryName}');

    // 카테고리 지출 내역 조회 API 호출
    return repository.getCategoryTransactions(
      budgetPk: params.budgetPk,
      categoryPk: params.categoryPk,
      startDate: params.startDate,
      endDate: params.endDate,
      categoryName: params.categoryName,
    );
  },
);

// Parameters class for the provider
class CategoryExpensePageParams {
  final int budgetPk;
  final int categoryPk;
  final String startDate;
  final String endDate;
  final String categoryName;

  CategoryExpensePageParams({
    required this.budgetPk,
    required this.categoryPk,
    required String rawStartDate,
    required String rawEndDate,
    required this.categoryName,
  })  : startDate = rawStartDate.replaceAll('-', ''),
        endDate = rawEndDate.replaceAll('-', '');

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is CategoryExpensePageParams &&
        other.budgetPk == budgetPk &&
        other.categoryPk == categoryPk &&
        other.startDate == startDate &&
        other.endDate == endDate &&
        other.categoryName == categoryName;
  }

  @override
  int get hashCode {
    return budgetPk.hashCode ^
        categoryPk.hashCode ^
        startDate.hashCode ^
        endDate.hashCode;
  }
}

/*
  예산 카테고리 지출 내역 페이지
  : 예산 카테고리의 상세 지출 내역을 보여주는 페이지입니다.
*/
class CategoryExpensePage extends ConsumerWidget {
  final int categoryPk;
  final BudgetCategoryModel category;
  final int budgetPk;

  const CategoryExpensePage({
    super.key,
    required this.categoryPk,
    required this.category,
    required this.budgetPk,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Get the budget to determine date range
    final budgetState = ref.watch(budgetProvider);

    // 안전하게 selectedBudget 찾기
    BudgetModel? selectedBudget;

    try {
      // 먼저 budgetPk로 찾기 시도
      if (budgetState.budgets.isNotEmpty) {
        selectedBudget = budgetState.budgets.firstWhere(
          (budget) => budget.budgetPk == budgetPk,
          orElse: () => throw Exception("Budget not found"),
        );
      }
    } catch (e) {
      // 찾지 못한 경우 selectedBudget 사용
      selectedBudget = budgetState.selectedBudget;
    }

    // 여전히 null인 경우 (예: budgets가 비어있고 selectedBudget도 null)
    // category 객체에서 날짜 데이터를 추출하거나 현재 날짜 사용
    if (selectedBudget == null) {
      // 현재 날짜를 기준으로 한 달 기간 설정
      final now = DateTime.now();
      final startDate = DateTime(now.year, now.month, 1);
      final endDate = DateTime(now.year, now.month + 1, 0);

      final params = CategoryExpensePageParams(
        budgetPk: budgetPk,
        categoryPk: categoryPk,
        rawStartDate: DateFormat('yyyy-MM-dd').format(startDate),
        rawEndDate: DateFormat('yyyy-MM-dd').format(endDate),
        categoryName: category.budgetCategoryName,
      );

      // 트랜잭션 데이터 로드
      final transactionsAsync = ref.watch(categoryTransactionsProvider(params));

      return _buildScaffold(context, ref, transactionsAsync, category, params);
    }

    // 정상적으로 예산 데이터를 찾은 경우
    final params = CategoryExpensePageParams(
      budgetPk: budgetPk,
      categoryPk: categoryPk,
      rawStartDate: selectedBudget.startDate,
      rawEndDate: selectedBudget.endDate,
      categoryName: category.budgetCategoryName,
    );

    // autoDispose 모디파이어 사용하여 불필요한 api 호출 방지
    final transactionsAsync = ref.watch(categoryTransactionsProvider(params));

    return _buildScaffold(context, ref, transactionsAsync, category, params);
  }

  // Scaffold 구성을 별도의 메서드로 분리
  Widget _buildScaffold(
      BuildContext context,
      WidgetRef ref,
      AsyncValue<List<Transaction>> transactionsAsync,
      BudgetCategoryModel category,
      CategoryExpensePageParams params) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: CustomAppbar(
        title: '지출 내역',
        backgroundColor: category.color.withOpacity(0.1),
      ),
      body: Column(
        children: [
          // 가계 지출 데이터를 기반으로 카테고리 요약 정보 업데이트
          transactionsAsync.when(
            data: (transactions) {
              // 지출 트랜잭션만 필터링하고 총 금액 계산
              final filteredTransactions = transactions
                  .where((transaction) =>
                          transaction.classification ==
                              TransactionClassification.WITHDRAWAL &&
                          transaction.exceptedBudgetYn != 'Y' // 예산 제외 아닌 것만 계산
                      )
                  .toList();

              // 총 지출액 계산
              final totalSpent = filteredTransactions.fold<int>(
                  0, (sum, transaction) => sum + transaction.householdAmount);

              // 예산 대비 지출 비율 계산
              final percentage = category.budgetCategoryPrice > 0
                  ? (totalSpent / category.budgetCategoryPrice * 100)
                  : 0.0;

              return _buildCategorySummary(
                  context, category, totalSpent, percentage);
            },
            loading: () => _buildCategorySummary(
              context,
              category,
              category.budgetExpendAmount ?? 0,
              category.budgetExpendPercent ?? 0,
            ),
            error: (_, __) => _buildCategorySummary(
                context,
                category,
                category.budgetExpendAmount ?? 0,
                category.budgetExpendPercent ?? 0),
          ),

          // Transaction List
          Expanded(
            child: transactionsAsync.when(
              data: (transactions) {
                if (transactions.isEmpty) {
                  return _buildEmptyState();
                }
                return _buildTransactionList(context, ref, transactions);
              },
              loading: () => Center(
                child: Lottie.asset(
                  'assets/images/loading/loading_simple.json',
                  width: 140,
                  height: 140,
                  fit: BoxFit.contain,
                ),
              ),
              error: (error, stackTrace) => Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.error_outline,
                        size: 48, color: AppColors.pinkPrimary),
                    const SizedBox(height: 16),
                    Text(
                      '오류가 발생했습니다',
                      style: AppTextStyles.bodyMedium,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      error.toString(),
                      style: AppTextStyles.bodyExtraSmall
                          .copyWith(color: AppColors.disabled),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton(
                      onPressed: () {
                        ref.refresh(categoryTransactionsProvider(params));
                      },
                      child: const Text('다시 시도'),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCategorySummary(
      BuildContext context,
      BudgetCategoryModel category,
      int actualSpent, // 예산별 가계부 조회로부터 가지고 온 카테고리별 지출금액
      double actualPercentage // 예산별 가계부 조회로부터 가지고 온 카테고리별 지출 비율(카테고리의 예산에 대한)
      ) {
    final formattedBudget = _formatAmount(category.budgetCategoryPrice);
    final formattedSpent = _formatAmount(actualSpent);
    final percentage = actualPercentage;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: category.color.withOpacity(0.1),
        borderRadius: const BorderRadius.only(
          bottomLeft: Radius.circular(16),
          bottomRight: Radius.circular(16),
        ),
      ),
      child: Column(
        children: [
          // Category Icon & Name
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: category.color.withOpacity(0.3),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  _getCategoryIcon(category.budgetCategoryName),
                  color: category.color,
                  size: 24,
                ),
              ),
              const SizedBox(width: 12),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    category.budgetCategoryName,
                    style: AppTextStyles.bodyLarge,
                  ),
                  Text(
                    percentage > 100 ? '예산 초과' : '예산 내에서 사용 중',
                    style: AppTextStyles.bodyExtraSmall.copyWith(
                      color: percentage > 100
                          ? AppColors.buttonDelete
                          : AppColors.backgroundBlack,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 16),

          // Progress Bar
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: LinearProgressIndicator(
              value: percentage / 100,
              backgroundColor: AppColors.whiteDark,
              valueColor: AlwaysStoppedAnimation<Color>(
                percentage > 100 ? AppColors.buttonDelete : category.color,
              ),
              minHeight: 8,
            ),
          ),
          const SizedBox(height: 12),

          // Budget vs Spent
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _buildAmountItem(
                  '사용금액',
                  '$formattedSpent원',
                  percentage > 100
                      ? AppColors.buttonDelete
                      : AppColors.textPrimary),
              _buildAmountItem(
                '사용률',
                '${percentage.toStringAsFixed(0)}%',
                percentage > 100 ? AppColors.buttonDelete : category.color,
              ),
              _buildAmountItem(
                  '예산금액', '$formattedBudget원', AppColors.textSecondary),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAmountItem(String label, String value, Color valueColor) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          label,
          style: AppTextStyles.bodyExtraSmall.copyWith(
            color: AppColors.textSecondary,
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

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.receipt_long_outlined,
            size: 64,
            color: AppColors.disabled,
          ),
          const SizedBox(height: 16),
          Text(
            '지출 내역이 없습니다',
            style: AppTextStyles.bodyMedium.copyWith(
              color: AppColors.textSecondary,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '카테고리의 지출 내역이 표시됩니다',
            style: AppTextStyles.bodySmall.copyWith(
              color: AppColors.disabled,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTransactionList(
      BuildContext context, WidgetRef ref, List<Transaction> transactions) {
    // 지출(WITHDRAWAL) 트랜잭션만 필터링
    final filteredTransactions = transactions
        .where((transaction) =>
            transaction.classification == TransactionClassification.WITHDRAWAL)
        .toList();

    if (filteredTransactions.isEmpty) {
      return _buildEmptyState();
    }

    // Group transactions by date
    final groupedTransactions = <String, List<Transaction>>{};
    for (final transaction in filteredTransactions) {
      final date = transaction.tradeDate;

      if (!groupedTransactions.containsKey(date)) {
        groupedTransactions[date] = [];
      }
      groupedTransactions[date]!.add(transaction);
    }

    // Sort dates in descending order
    final sortedDates = groupedTransactions.keys.toList()
      ..sort((a, b) => b.compareTo(a));

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: sortedDates.length,
      itemBuilder: (context, index) {
        final date = sortedDates[index];
        final dayTransactions = groupedTransactions[date]!;

        // tradeDate : 20250409 형식으로 들어온다
        DateTime dateTime;
        try {
          final year = int.parse(date.substring(0, 4));
          final month = int.parse(date.substring(4, 6));
          final day = int.parse(date.substring(6, 8));
          dateTime = DateTime(year, month, day);
        } catch (e) {
          // 날짜 형식이 유효하지 않은 경우 현재 날짜 사용
          dateTime = DateTime.now();
        }

        final displayFormat = DateFormat('yyyy년 MM월 dd일 (E)', 'ko_KR');
        final displayDate = displayFormat.format(dateTime);

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Date header
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: Text(
                displayDate,
                style: AppTextStyles.bodyMedium.copyWith(
                  fontWeight: FontWeight.w300,
                ),
              ),
            ),

            // Transactions for this date
            ...dayTransactions.map((transaction) =>
                _buildTransactionItem(context, ref, transaction)),

            const SizedBox(height: 8),
            const Divider(
              thickness: 0.5,
            ),
          ],
        );
      },
    );
  }

  Widget _buildTransactionItem(
      BuildContext context, WidgetRef ref, Transaction transaction) {
    final formattedAmount = _formatAmount(transaction.householdAmount);

    // Format time
    String displayTime = "";
    try {
      String timeStr = transaction.tradeTime;

      final hours = int.parse(timeStr.substring(0, 2));
      final minutes = int.parse(timeStr.substring(2, 4));

      final time = DateTime(2000, 1, 1, hours, minutes);
      displayTime = DateFormat('HH:mm').format(time);
    } catch (e) {
      // 시간 형식이 유효하지 않은 경우 원본 문자열 사용
      displayTime = transaction.tradeTime;
    }

    return GestureDetector(
      onTap: () {
        // Show transaction detail modal
        showCustomModal(
          context: context,
          ref: ref,
          backgroundColor: AppColors.background,
          child: TransactionDetailModal(
            householdPk: transaction.householdPk,
            readOnly: true,
          ),
        );
      },
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: AppColors.backgroundBlack,
            width: 0.5,
          ),
        ),
        child: Row(
          children: [
            // Transaction icon or image
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: category.color.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.store,
                color: category.color,
                size: 20,
              ),
            ),
            const SizedBox(width: 12),

            // Transaction details
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        transaction.tradeName,
                        style: AppTextStyles.bodyMedium,
                      ),
                      Text(
                        '$formattedAmount원',
                        style: AppTextStyles.bodyMedium.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        transaction.householdMemo ?? '',
                        style: AppTextStyles.bodyExtraSmall.copyWith(
                          color: AppColors.textSecondary,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      Row(
                        children: [
                          Text(
                            displayTime,
                            style: AppTextStyles.bodyExtraSmall.copyWith(
                              color: AppColors.textSecondary,
                            ),
                          ),
                          if (transaction.exceptedBudgetYn == 'Y')
                            Container(
                              margin: const EdgeInsets.only(left: 4),
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 4, vertical: 2),
                              decoration: BoxDecoration(
                                color: AppColors.yellowLight,
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: Text(
                                '예산제외',
                                style: AppTextStyles.bodyExtraSmall.copyWith(
                                  color: AppColors.yellowDark,
                                  fontSize: 10,
                                ),
                              ),
                            ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  IconData _getCategoryIcon(String categoryName) {
    switch (categoryName) {
      case '식비':
        return Icons.restaurant;
      case '교통비':
        return Icons.directions_bus;
      case '여가':
        return Icons.sports_esports;
      case '커피/디저트':
        return Icons.coffee;
      case '쇼핑':
        return Icons.shopping_bag;
      case '생활':
        return Icons.home;
      case '주거':
        return Icons.house;
      case '의료':
        return Icons.medical_services;
      default:
        return Icons.money;
    }
  }

  String _formatAmount(int amount) {
    return amount.toString().replaceAllMapped(
        RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'), (Match m) => '${m[1]},');
  }
}
