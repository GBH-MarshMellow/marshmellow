import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

/*
  커스텀 버튼 위젯
*/
class CustomButton extends StatelessWidget {
  final String text;
  final VoidCallback? onPressed;
  final bool isEnabled;

  const CustomButton({
    required this.text,
    required this.onPressed,
    this.isEnabled = true,
    Key? key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {

    final double screenWidth = MediaQuery.of(context).size.width;

    return Center(
      child: SizedBox(
        width: screenWidth * 0.9,
        height: 60,
        child: ElevatedButton(
          onPressed: isEnabled ? onPressed : null,
          style: ElevatedButton.styleFrom(
            backgroundColor: isEnabled ? AppColors.backgroundBlack : AppColors.textSecondary.withOpacity(0.5),
            foregroundColor: AppColors.whiteLight,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(5),
            ),
            elevation: 0,
          ),
          child: Text(
            text,
            style: AppTextStyles.bodySmall.copyWith(
              color: AppColors.whiteLight,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ),
    );
  }
}