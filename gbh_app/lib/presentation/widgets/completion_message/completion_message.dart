import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:flutter/material.dart';

class CompletionMessage extends StatelessWidget {
  final String message;

  const CompletionMessage({
    super.key,
    required this.message,
  });

  static void show(BuildContext context, {required String message}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Container(
          padding: const EdgeInsets.symmetric(horizontal: 5),
          height: 20,
          child: Text(
            message,
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: AppColors.background,
              fontSize: 14,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
        backgroundColor: AppColors.textPrimary,
        elevation: 0,
        duration: const Duration(seconds: 2),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(50),
        ),
        margin: const EdgeInsets.only(
          bottom: 20,
          left: 130,
          right: 130,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return const SizedBox.shrink();
  }
}
