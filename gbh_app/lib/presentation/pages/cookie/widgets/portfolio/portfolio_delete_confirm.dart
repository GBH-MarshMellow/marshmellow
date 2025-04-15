import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';

class ConfirmModal extends StatelessWidget {
  final String title;
  final String subtitle;
  final String description;
  final String cancelText;
  final String confirmText;
  final VoidCallback? onCancel;
  final VoidCallback? onConfirm;

  const ConfirmModal({
    super.key,
    required this.title,
    required this.subtitle,
    required this.description,
    this.cancelText = '취소',
    this.confirmText = '삭제',
    this.onCancel,
    this.onConfirm,
  });

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: AppColors.whiteLight,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      content: SizedBox(
        width: 300,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              title,
              style: AppTextStyles.bodySmall.copyWith(
                color: AppColors.warnning,
                fontWeight: FontWeight.normal,
              ),
            ),
            const SizedBox(height: 8),
            Text(subtitle,
                textAlign: TextAlign.center,
                style: AppTextStyles.bodySmall
                    .copyWith(fontWeight: FontWeight.w400)),
            const SizedBox(height: 15),
            Text(
              description,
              textAlign: TextAlign.center,
              style: AppTextStyles.bodySmall.copyWith(
                fontWeight: FontWeight.w300,
              ),
            ),
            const SizedBox(height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Expanded(
                  child: Button(
                    onPressed: () {
                      if (onCancel != null) {
                        onCancel!();
                      } else {
                        Navigator.of(context).pop(false);
                      }
                    },
                    text: cancelText,
                    isDisabled: false,
                    color: AppColors.disabled,
                    textStyle: AppTextStyles.bodyMedium.copyWith(
                      color: AppColors.whiteLight,
                      fontSize: 14,
                    ),
                    width: double.infinity,
                    height: 40,
                    borderColor: AppColors.greyLight,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Button(
                    onPressed: () {
                      if (onConfirm != null) {
                        onConfirm!();
                      }
                      Navigator.of(context).pop(true);
                    },
                    text: confirmText,
                    isDisabled: false,
                    color: AppColors.warnning,
                    textStyle: AppTextStyles.bodyMedium.copyWith(
                      color: Colors.white,
                      fontSize: 14,
                    ),
                    width: double.infinity,
                    height: 40,
                  ),
                )
              ],
            ),
          ],
        ),
      ),
    );
  }
}
