import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class CustomComplete extends StatelessWidget {
  final Color backgroundColor;
  final String message;
  static String wormPath = 'assets/images/mm/mm_worm_bk.svg';

  const CustomComplete({
    Key? key,
    required this.backgroundColor,
    required this.message,
  }) : super(key:key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: backgroundColor,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SvgPicture.asset(
              wormPath,
              width: 80,
              height: 80,
            ),
            const SizedBox(height: 20),
            Text(
              message,
              style: AppTextStyles.bodyLarge,
              textAlign: TextAlign.center,
            )
          ],
        ),
      ),
    );
  }
}