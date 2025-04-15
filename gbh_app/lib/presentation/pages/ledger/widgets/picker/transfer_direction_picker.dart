import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/category/transfer_category.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/transfer_category_picker.dart';

// 이체 방향 선택 열거형
enum TransferDirection {
  deposit, // 입금
  withdrawal, // 출금
}

// 이체 방향 선택 모달
class TransferDirectionPicker extends StatelessWidget {
  const TransferDirectionPicker({Key? key}) : super(key: key);

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
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
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
                Text(
                  '카테고리',
                  style: AppTextStyles.bodyMedium.copyWith(
                    fontWeight: FontWeight.w300,
                  ),
                ),
              ],
            ),
          ),

          // 방향 선택 버튼
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildDirectionButton(
                  context: context,
                  label: '입금',
                  iconPath: IconPath.paid,
                  direction: TransferDirection.deposit,
                ),
                _buildDirectionButton(
                  context: context,
                  label: '출금',
                  iconPath: IconPath.received,
                  direction: TransferDirection.withdrawal,
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  // 방향 선택 버튼 위젯
  Widget _buildDirectionButton({
    required BuildContext context,
    required String label,
    required String iconPath,
    required TransferDirection direction,
  }) {
    return InkWell(
      onTap: () async {
        // 방향 선택 모달 닫지 않기

        // 카테고리 선택 모달 띄우기
        final selectedCategory = await showTransferCategoryPickerModal(
          context,
          direction: direction,
        );

        // 선택된 카테고리가 있으면 결과 반환
        if (selectedCategory != null && context.mounted) {
          Navigator.pop(context, {
            'direction': direction,
            'category': selectedCategory,
          });
        }
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade200),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Row(
          children: [
            SvgPicture.asset(
              iconPath,
              width: 18,
              height: 18,
              colorFilter:
                  ColorFilter.mode(Colors.grey.shade700, BlendMode.srcIn),
            ),
            const SizedBox(width: 10),
            Text(
              label,
              style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
      ),
    );
  }
}

// 이체 방향 선택 모달 표시 함수
Future<Map<String, dynamic>?> showTransferDirectionPickerModal(
    BuildContext context) async {
  return await showModalBottomSheet<Map<String, dynamic>>(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (context) => const TransferDirectionPicker(),
  );
}

// 이체 카테고리 선택 모달 표시 함수
Future<TransferCategory?> showTransferCategoryPickerModal(
  BuildContext context, {
  required TransferDirection direction,
}) async {
  return await showModalBottomSheet<TransferCategory>(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (context) => TransferCategoryPicker(
      direction: direction,
    ),
  );
}
