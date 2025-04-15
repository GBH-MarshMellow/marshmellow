import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/di/providers/auth/certificate_process_provider.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/custom_button.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/text_input/text_input.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

/*
  mm인증서 회원가입시 이메일 provider
*/
final emailProvider = StateProvider<String>((ref) => '');

class AuthMydataEmailInputPage extends ConsumerWidget {
  const AuthMydataEmailInputPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final screenWidth = MediaQuery.of(context).size.width;
    final email = ref.watch(emailProvider);
    final isValidEmail = _isValidEmail(email);

    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(40.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 20),
            const Text('이메일 입력', style: AppTextStyles.mainTitle),
            const SizedBox(height: 10),
            Text(
              'MM 인증서 발급을 위해 이메일을 입력해 주세요',
              style: AppTextStyles.bodySmall.copyWith(color: AppColors.disabled),
            ),
            const SizedBox(height: 40),

            // 이메일 입력 필드
            EmailInputSection(),

            // 이메일 유효성 검사 메시지
            if (email.isNotEmpty && !isValidEmail)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Text(
                  '올바른 이메일 형식이 아닙니다',
                  style: AppTextStyles.bodySmall,
                ),
              ),
              const Spacer(),

              // 다음 버튼
              Button(
                text: '다음',
                width: screenWidth * 0.9,
                height: 60,
                onPressed: isValidEmail ? () {
                  // 이메일을 certificateprocessprovider 에 저장
                  ref.read(certificateProcessProvider.notifier).setEmail(email);
                  
                  Navigator.of(context).pop();
                  context.go(SignupRoutes.getMyDataPasswordPath());
                  
                } : null,
              ),
              const SizedBox(height: 20),
          ],
        ),
      )
    );
  }

  // 이메일 유효성 검사
  bool _isValidEmail(String email) {
    if (email.isEmpty) return false;

    final emailRegex = RegExp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$');
    return emailRegex.hasMatch(email);
  }
}

class EmailInputSection extends ConsumerWidget {
  final TextEditingController _emailController = TextEditingController();

  EmailInputSection({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final email = ref.watch(emailProvider);

    // 컨트롤러 동기화
    if (_emailController.text != email) {
      _emailController.text = email;
      // 커서 위치 유지
      _emailController.selection = TextSelection.fromPosition(
        TextPosition(offset: _emailController.text.length),
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextInput(
          label: '이메일',
          controller: _emailController,
          onChanged: (value) => ref.read(emailProvider.notifier).state = value,
        )
      ],
    );
  }
}