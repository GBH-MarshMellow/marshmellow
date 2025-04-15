import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/src/widgets/framework.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/certification_card.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

/*
  mm인증서 선택 모달 UI
*/
class AuthMydataCertSelectModal extends ConsumerWidget {
  final String userName;
  final VoidCallback? onDismiss;

  const AuthMydataCertSelectModal({
    Key? key,
    required this.userName,
    this.onDismiss,
  }) : super(key: key);

  static Future<void> show(BuildContext context, String userName) {
    return showModalBottomSheet(
      context: context, 
      isScrollControlled: true,
      backgroundColor: AppColors.whiteLight,
      builder: (context) => AuthMydataCertSelectModal(
        userName: userName, 
        onDismiss: () => Navigator.of(context).pop(),
      ),
    );
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final screenWidth = MediaQuery.of(context).size.width;

    return BackdropFilter(
      filter: ImageFilter.blur(sigmaX: 5, sigmaY: 5),
      child: Container(
        decoration: const BoxDecoration(
          color: AppColors.whiteLight,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(24),
            topRight: Radius.circular(24),
          ),
        ),
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '인증서 로그인을 해주세요',
              style: AppTextStyles.bodyMedium,
            ),
            const SizedBox(height: 24),

            Center(
              child: SmallCertificateCard(
                userName: userName, 
                expiryDate: '2028.03.14',
                onTap: () {
                  // 인증서를 탭하면 무엇이 어떻게 변할 계획이죠?
                },
              ),
            ),
            const SizedBox(height: 32),

            Button(
              text: '다음',
              width: screenWidth * 0.9,
              height: 60,
              onPressed: () {
                Navigator.of(context).pop();
                context.go(SignupRoutes.getMyDataLoginPath());
              },
              isDisabled: false,
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }
}

extension AuthMydataCertSelectExtension on BuildContext {
  Future<void> showAuthMydataCertSelect(String userName) {
    return AuthMydataCertSelectModal.show(this, userName);
  }
}