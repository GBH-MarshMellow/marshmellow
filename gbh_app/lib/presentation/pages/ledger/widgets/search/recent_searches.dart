import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class RecentSearches extends StatelessWidget {
  final List<String> searches;
  final VoidCallback onClearAll;
  final Function(String) onSearchTermSelected;
  final Function(String) onSearchTermDeleted;

  const RecentSearches({
    super.key,
    required this.searches,
    required this.onClearAll,
    required this.onSearchTermSelected,
    required this.onSearchTermDeleted,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              '최근 검색어',
              style:
                  AppTextStyles.bodySmall.copyWith(fontWeight: FontWeight.w300),
            ),
            TextButton(
              onPressed: onClearAll,
              style: TextButton.styleFrom(
                padding: EdgeInsets.zero, // 패딩 제거
                minimumSize: Size.zero, // 최소 크기 제거
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
              child: Text(
                '전체 삭제',
                style: AppTextStyles.bodySmall
                    .copyWith(fontWeight: FontWeight.w300),
              ),
            ),
          ],
        ),
        const Divider(
          color: AppColors.textSecondary,
          thickness: 0.5,
        ),
        const SizedBox(height: 10),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: searches.map((term) => _buildSearchChip(term)).toList(),
        ),
      ],
    );
  }

  Widget _buildSearchChip(String term) {
    return GestureDetector(
      onTap: () => onSearchTermSelected(term),
      child: Chip(
        label: Text(term, style: AppTextStyles.bodyMedium),
        deleteIcon: Icon(Icons.close, size: 16, color: AppColors.greyPrimary),
        onDeleted: () => onSearchTermDeleted(term),
        backgroundColor: AppColors.whiteLight,
        deleteButtonTooltipMessage: '삭제',
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(5),
          side: BorderSide(color: AppColors.textSecondary),
        ),
      ),
    );
  }
}
