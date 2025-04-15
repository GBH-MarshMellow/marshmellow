import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/category/deposit_category.dart';

class IncomeCategoryPicker extends StatelessWidget {
  final Function(DepositCategory) onCategorySelected;
  final String? title;
  final bool showCloseButton;

  const IncomeCategoryPicker({
    Key? key,
    required this.onCategorySelected,
    this.title = '수입 카테고리',
    this.showCloseButton = false,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(10),
          topRight: Radius.circular(10),
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Header
          Container(
            padding: const EdgeInsets.symmetric(vertical: 16),
            decoration: BoxDecoration(
              border: Border(
                bottom: BorderSide(
                  color: Colors.grey.shade200,
                  width: 1,
                ),
              ),
            ),
            child: Row(
              children: [
                const SizedBox(width: 20),
                Text(
                  title!,
                  style: AppTextStyles.bodyMedium.copyWith(
                    fontWeight: FontWeight.w300,
                  ),
                ),
              ],
            ),
          ),

          // Categories Grid
          Flexible(
            child: GridView.builder(
              shrinkWrap: true,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                childAspectRatio: 3.0,
                crossAxisSpacing: 8,
                mainAxisSpacing: 8,
              ),
              itemCount: DepositCategory.allCategories.length,
              itemBuilder: (context, index) {
                final category = DepositCategory.allCategories[index];
                return _CategoryTile(
                  category: category,
                  onTap: () {
                    onCategorySelected(category);
                    Navigator.of(context).pop();
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

// CategoryTile 클래스
class _CategoryTile extends StatelessWidget {
  final DepositCategory category;
  final VoidCallback onTap;

  const _CategoryTile({
    Key? key,
    required this.category,
    required this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade200),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Row(
          children: [
            const SizedBox(width: 16),
            SizedBox(
              width: 24,
              height: 24,
              child: SvgPicture.asset(
                category.iconPath,
                colorFilter: ColorFilter.mode(Colors.grey.shade700,
                    BlendMode.srcIn), // color 대신 colorFilter 사용
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                category.name,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                ),
                overflow: TextOverflow.ellipsis, // 오타 수정
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// showCategoryPickerModal 함수 - 카테고리 선택기 모달을 표시
Future<void> showCategoryPickerModal(
  BuildContext context, {
  required Function(DepositCategory) onCategorySelected,
}) {
  return showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (context) => IncomeCategoryPicker(
      onCategorySelected: onCategorySelected,
      showCloseButton: false,
    ),
  );
}
