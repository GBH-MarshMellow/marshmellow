/*
  mm인증서 로그인 UI
*/
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/di/providers/auth/certificate_process_provider.dart';
import 'package:marshmellow/di/providers/auth/mydata_provider.dart';
import 'package:marshmellow/presentation/widgets/dots_input/dots_input.dart';
import 'package:marshmellow/presentation/widgets/keyboard/index.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

class AuthMydataCertLoginPage extends ConsumerWidget {
  const AuthMydataCertLoginPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final loginState = ref.watch(MydataLoginProvider);
    final certProcessState = ref.watch(certificateProcessProvider);

    void showKeyboard() {
      KeyboardModal.showSecureNumericKeyboard(
        context: context, 
        onValueChanged: (value) {
          final notifier = ref.read(MydataLoginProvider.notifier);
          notifier.resetPassword();
          // 각자리 순차적으로 입력
          for (int i = 0; i < value.length && i < 6; i++) {
            notifier.addDigit(value[i]);
          }

          // 비밀번호 6자리 되면 자동으로 로그인 시도
          if (value.length == 6) {
            Future.delayed(const Duration(milliseconds: 300), () async {
              // 프로세스 프로바이더에 비밀번호 설정
              ref.read(certificateProcessProvider.notifier).setPassword(value);

              // 로그인 시도
              final success = await notifier.loginWithCertificate(value);
              if (success) {
                // 로그인 성공시
                context.go(SignupRoutes.getMyDataAgreementPath());
              } else {
                // 로그인 실패시
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('비밀번호가 일치하지 않습니다'),
                    duration: Duration(seconds: 2),
                  )
                );
                // 입력값 초기화
                notifier.resetPassword();
              }
            });
          }
        }, 
        initialValue: loginState.password,
        maxLength: 6,
      );
    }
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 40.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 60),
              Text(
                'MM 인증서 로그인\n비밀번호 6자리를 입력해 주세요',
                style: AppTextStyles.bodyLarge,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 50),
              PinDotsRow(
                currentDigit: loginState.currentDigit, 
                onTap: showKeyboard,
              ),
              const Spacer(),
              if (loginState.isLoading || certProcessState.isLoading)
                const Center(
                  child: CircularProgressIndicator(),
                ),

              // 에러메시지 표시
              if (loginState.error != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 20.0),
                  child: Text(
                    loginState.error!,
                    style: TextStyle(color: AppColors.buttonDelete),
                    textAlign: TextAlign.center,
                  ),
                ),
            ],
          )
        )
      ),
    );
  }
}

