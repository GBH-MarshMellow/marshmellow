import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/di/providers/auth/pin_provider.dart';

/*
  생체인식 사용 여부를 확인 UI
*/
class BiometricOption extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return TextButton.icon(
      icon: const Icon(Icons.fingerprint, color: AppColors.greyPrimary),
      label: Text('다음부터 Face ID 쓰기', style: AppTextStyles.bodyExtraSmall.copyWith(color: AppColors.blackLight)),
      onPressed: () {
        ref.read(pinStateProvider.notifier).toggleBiometrics(true);
      }, 
    );
  }
}