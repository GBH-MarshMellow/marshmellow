import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class TypeSelector extends StatelessWidget {
  final String selectedType;
  final Function(String) onTypeSelected;
  final bool enabled;

  const TypeSelector({
    super.key,
    required this.selectedType,
    required this.onTypeSelected,
    this.enabled = true,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        // 수입 버튼
        Expanded(
          child: _buildTypeButton('수입', selectedType == '수입'),
        ),
        const SizedBox(width: 8),

        // 지출 버튼
        Expanded(
          child: _buildTypeButton('지출', selectedType == '지출'),
        ),
        const SizedBox(width: 8),

        // 이체 버튼
        Expanded(
          child: _buildTypeButton('이체', selectedType == '이체'),
        ),
      ],
    );
  }

  Widget _buildTypeButton(String type, bool isSelected) {
    // 타입별 활성화 색상
    Color getActiveColor() {
      switch (type) {
        case '수입':
          return AppColors.bluePrimary;
        case '지출':
          return AppColors.pinkPrimary;
        case '이체':
          return AppColors.yellowPrimary;
        default:
          return Colors.transparent;
      }
    }

    // 타입별 비활성화 색상
    Color getDisabledColor() {
      switch (type) {
        case '수입':
          return AppColors.blueLight;
        case '지출':
          return AppColors.pinkLight;
        case '이체':
          return AppColors.yellowLight;
        default:
          return Colors.transparent;
      }
    }

    // 버튼 배경색 결정
    final buttonColor = !enabled && isSelected
        ? getDisabledColor() // 비활성화 상태에서 선택된 경우
        : isSelected
            ? getActiveColor() // 활성화 상태에서 선택된 경우
            : Colors.transparent; // 선택되지 않은 경우

    // 테두리 색상 결정
    final borderColor = !enabled && isSelected
        ? AppColors.textLight
        : isSelected
            ? getActiveColor() // 활성화 상태에서 선택된 경우
            : AppColors.textLight; // 선택되지 않은 경우

    // 텍스트 색상 결정
    final textColor = !enabled
        ? AppColors.textLight // 비활성화 상태
        : isSelected
            ? AppColors.textPrimary // 활성화 상태에서 선택된 경우
            : AppColors.textLight; // 선택되지 않은 경우

    return GestureDetector(
      onTap: enabled ? () => onTypeSelected(type) : null,
      child: Container(
        height: 32,
        width: 40,
        decoration: BoxDecoration(
          color: buttonColor,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: borderColor,
            width: 1,
          ),
        ),
        alignment: Alignment.center,
        child: Text(
          type,
          style: AppTextStyles.bodySmall.copyWith(
            color: textColor,
            fontWeight: isSelected ? FontWeight.w500 : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}
