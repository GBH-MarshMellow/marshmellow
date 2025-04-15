import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';
import 'package:go_router/go_router.dart';

class AuthVerificationLoadingPage extends StatefulWidget {
  const AuthVerificationLoadingPage({Key? key}) : super(key: key);

  @override
  State<AuthVerificationLoadingPage> createState() => _AuthVerificationLoadingPageState();
}

class _AuthVerificationLoadingPageState extends State<AuthVerificationLoadingPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: const Center(
        child: CustomLoadingIndicator(text:'본인인증을\n확인하고 있습니다',backgroundColor: AppColors.whiteLight, opacity: 0.5,),
      ),
      // 일단 임시로 넘어갈건데 여기서 본인인증 기다리고 있으세요!
      floatingActionButton: FloatingActionButton(
       onPressed: () {
        context.go(SignupRoutes.getAuthCompletePath());
       }
      ),
    );
  }
}

