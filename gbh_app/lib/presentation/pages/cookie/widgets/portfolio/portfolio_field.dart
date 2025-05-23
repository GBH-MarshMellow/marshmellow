import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class PortfolioField extends StatelessWidget {
  final String label;
  final String? value;
  final VoidCallback? onTap;
  final Widget? trailing;
  final bool? showDivider;
  final EdgeInsetsGeometry? padding;
  final MainAxisAlignment rowAlignment;
  final double labelWidth;
  final TextStyle? valueStyle;

  const PortfolioField({
    super.key,
    required this.label,
    this.value,
    this.onTap,
    this.trailing,
    this.showDivider = true,
    this.padding = const EdgeInsets.symmetric(horizontal: 10, vertical: 17),
    this.rowAlignment = MainAxisAlignment.spaceBetween,
    this.labelWidth = 100,
    this.valueStyle,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        InkWell(
          onTap: onTap,
          child: Padding(
            padding: padding!,
            child: Row(
              mainAxisAlignment: rowAlignment,
              children: [
                // 라벨 텍스트 (고정 너비)
                SizedBox(
                  width: labelWidth,
                  child: Text(
                    label,
                    style: AppTextStyles.bodySmall
                        .copyWith(color: AppColors.textSecondary),
                  ),
                ),

                // 오른쪽 콘텐츠 (나머지 공간 사용)
                Expanded(
                  child: Align(
                    alignment: Alignment.centerRight,
                    child: trailing ?? _buildDefaultTrailing(),
                  ),
                ),
              ],
            ),
          ),
        ),
        if (showDivider!)
          const Divider(height: 0.5, color: AppColors.textLight),
      ],
    );
  }

  Widget _buildDefaultTrailing() {
    return Text(
      value ?? '',
      style: valueStyle ??
          AppTextStyles.bodySmall.copyWith(color: AppColors.textPrimary),
    );
  }
}
