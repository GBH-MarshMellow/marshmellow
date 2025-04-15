import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_category_model.dart';
import 'package:marshmellow/presentation/viewmodels/portfolio/portfolio_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/portfolio/portfolio_category_viewmodel.dart';

class PortfolioCategoryPicker extends ConsumerWidget {
  final String selectedCategory;
  final Function(String, int) onCategorySelected;

  const PortfolioCategoryPicker({
    super.key,
    required this.selectedCategory,
    required this.onCategorySelected,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // 디버깅 로그 추가
    print('PortfolioCategoryPicker 빌드: 선택된 카테고리 = "$selectedCategory"');

    final categoryState = ref.watch(portfolioCategoryViewModelProvider);
    final categories = categoryState.categories;

    // 카테고리 목록 디버깅
    print('카테고리 개수: ${categories.length}');
    for (var cat in categories) {
      print(
          '카테고리: ${cat.portfolioCategoryName}, PK: ${cat.portfolioCategoryPk}');
    }

    return Container(
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.only(
          topLeft: Radius.circular(10),
          topRight: Radius.circular(10),
        ),
      ),
      // 최대 높이 제한 추가
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.7,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // 카테고리 목록 또는 로딩 표시기
          categoryState.isLoading
              ? const Padding(
                  padding: EdgeInsets.all(20.0),
                  child: Center(child: CircularProgressIndicator()),
                )
              : categories.isEmpty
                  ? _buildEmptyState()
                  : _buildCategoryList(context, categories),

          // 하단 여백
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  // 카테고리 없을 때 표시할 상태
  Widget _buildEmptyState() {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(height: 20),
            Text(
              '카테고리가 없습니다.',
              style: AppTextStyles.bodyMedium,
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  // 카테고리 목록 위젯
  Widget _buildCategoryList(
      BuildContext context, List<PortfolioCategoryModel> categories) {
    final filteredCategories = categories
        .where((category) => category.portfolioCategoryName != '미분류')
        .toList();
    return Flexible(
      child: ListView.builder(
        shrinkWrap: true,
        padding: const EdgeInsets.symmetric(vertical: 8),
        itemCount: filteredCategories.length,
        itemBuilder: (context, index) {
          final category = filteredCategories[index];
          final isSelected = category.portfolioCategoryName == selectedCategory;

          return InkWell(
            onTap: () {
              print(
                  '카테고리 선택됨: ${category.portfolioCategoryName}, ${category.portfolioCategoryPk}');
              onCategorySelected(
                  category.portfolioCategoryName, category.portfolioCategoryPk);
              Navigator.pop(context);
            },
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 12),
              child: Center(
                child: Text(
                  category.portfolioCategoryName,
                  style: AppTextStyles.bodyMedium.copyWith(
                    fontWeight:
                        isSelected ? FontWeight.bold : FontWeight.normal,
                    color: isSelected
                        ? AppColors.textLight
                        : AppColors.textPrimary,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
