import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/category/transfer_category.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/transfer_direction_picker.dart';

class TransferCategoryPicker extends StatelessWidget {
  final TransferDirection direction;
  final String? title;

  const TransferCategoryPicker({
    Key? key,
    required this.direction,
    this.title,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // 방향에 따른 제목 설정
    final modalTitle = title ?? (direction == TransferDirection.deposit ? '입금 카테고리' : '출금 카테고리');
    
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
          // 핸들바
          Center(
            child: Container(
              margin: const EdgeInsets.only(top: 12, bottom: 8),
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.grey.shade300,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
          ),
          
          // 헤더
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
                  modalTitle,
                  style: AppTextStyles.bodyMedium.copyWith(
                    fontWeight: FontWeight.w300,
                  ),
                ),
              ],
            ),
          ),

          // 카테고리 그리드
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
              itemCount: TransferCategory.allCategories.length,
              itemBuilder: (context, index) {
                final category = TransferCategory.allCategories[index];
                return _CategoryTile(
                  category: category,
                  onTap: () {
                    // 카테고리 선택 시 바로 결과 반환 및 모달 닫기
                    Navigator.pop(context, category);
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
  final TransferCategory category;
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
                colorFilter:
                    ColorFilter.mode(Colors.grey.shade700, BlendMode.srcIn),
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
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ),
    );
  }
}