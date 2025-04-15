import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/custom_button.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

class AuthCompletePage extends StatelessWidget {
  const AuthCompletePage({Key? key}) : super(key: key);
  

  @override
  Widget build(BuildContext context) {

    final double screenWidth = MediaQuery.of(context).size.width;

    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(40.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const SizedBox(height: 150,),
            Icon(
              Icons.check_sharp,
              size: 100,
              color: AppColors.blueDark,
            ),
            const SizedBox(height: 40),
            Text(
              '본인인증이 완료되었습니다',
              style: AppTextStyles.bodyLarge,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 300),
            Button(
              text:'시작하기',
              width: screenWidth * 0.9,
              height: 60,
              onPressed: () {
                context.go(SignupRoutes.getPinSetupPath());
              },
            )
          ],
        ),
      ),
    );
  }
}