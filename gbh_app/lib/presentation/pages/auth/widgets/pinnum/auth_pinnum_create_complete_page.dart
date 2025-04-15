import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/custom_complete.dart';
import 'package:marshmellow/presentation/viewmodels/auth/certificate_notifier.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

/*
  핀넘버 설정 성공 UI
*/
class AuthPinnumCompletePage extends ConsumerStatefulWidget {
  const AuthPinnumCompletePage({Key? key}) : super(key: key);

  @override
  ConsumerState<AuthPinnumCompletePage> createState() => _AuthPinnumCompletePageState();
}

class _AuthPinnumCompletePageState extends ConsumerState<AuthPinnumCompletePage> {

  @override
  void initState() {
    super.initState();
    // 컴포넌트가 마운트되면 통합인증 상태 확인
    _checkIntegratedAuthAndNavigate();
  }

  // 통합인증 상태 확인 및 라우팅 분개
  Future<void> _checkIntegratedAuthAndNavigate() async {
    // 핀번호 설정 성공 화면을 보여주기 위한 딜레이
    await Future.delayed(const Duration(seconds: 2));

    // 통합인증 상태 확인
    await ref.read(certificateProvider.notifier).checkCertificateStatus();

    final hasIntegratedAuth = ref.read(certificateProvider).hasIntegratedAuth;

    // 위젯 마운트된 상태인지 확인
    if (!mounted) return;

    if (hasIntegratedAuth) {
      // 통합인증된(서버에 유저키가 있다) 사용자는 앱메인페이지(예산) 이동
      context.go('/budget');
    } else {
      // 통합인증안된 사용자는 마이데이터 페이지 이동
      context.go(SignupRoutes.getMyDataSplashPath());
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: buildCompleteWidget(),
    );
  }

  Widget buildCompleteWidget() {
    return const CustomComplete(
      backgroundColor: AppColors.bluePrimary, 
      message: 'PIN 번호 설정 완료!'
    );
  }
}