import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/custom_complete.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

/*
  이미 마이데이터 연동된 사용자 확인 페이지
*/
class AuthAlreadyConnectedPage extends StatelessWidget {
  const AuthAlreadyConnectedPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {

    Future.delayed(const Duration(seconds: 2), () {
      context.go(SignupRoutes.getMyDataSplashPath());
    });
    return MaterialApp(
      home: buildBlueWidget(),
    );
  }

  Widget buildBlueWidget() {
    return const CustomComplete(
      backgroundColor: AppColors.yellowPrimary, 
      message: '당신은 이미 연동된 사용자입니다!\n메인페이지로 이동할게요'
    );
  }
}