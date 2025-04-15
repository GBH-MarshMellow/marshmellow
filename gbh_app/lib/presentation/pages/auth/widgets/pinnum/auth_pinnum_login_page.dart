import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/auth/pin_state.dart';
import 'package:marshmellow/di/providers/auth/pin_provider.dart';
import 'package:marshmellow/di/providers/auth/user_provider.dart';
import 'package:marshmellow/di/providers/core_providers.dart';
import 'package:marshmellow/presentation/widgets/dots_input/dots_input.dart';
import 'package:marshmellow/presentation/widgets/keyboard/index.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

class AuthPinnumLoginPage extends ConsumerWidget{
  const AuthPinnumLoginPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final pinState = ref.watch(pinStateProvider);
    final userState = ref.watch(userStateProvider);

    // 생체인식으로 로그인 시도
    void _tryBiometricAuth() async {
      if (pinState.isBiometricsAvailable && pinState.useBiometrics) {
        final success = await ref.read(pinStateProvider.notifier).loginWithBiometrics();
        
        if (success) {
          // 로그인 성공 후 메인 페이지로 이동
          context.go('/budget');
        } else {
          // 실패 시 에러 메시지
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('생체인식 로그인에 실패했습니다. PIN 번호로 로그인해주세요.'),
              backgroundColor: AppColors.buttonDelete,
            ),
          );
        }
      }
    }

    // PIN 입력 처리
    void handlePinInput() {
      KeyboardModal.showSecureNumericKeyboard(
        context: context,
        onValueChanged: (value) {
          // 입력값이 바뀔 때마다 프로바이더 호출
          final notifier = ref.read(pinStateProvider.notifier);

          // PIN 초기화
          notifier.resetPin();

          // 각자리 순차적으로 입력
          for (int i = 0; i < value.length && i < 4; i++) {
            notifier.addDigit(value[i]);
          }

          // PIN 4자리 되면 로그인 시도
          if (value.length == 4) {
            _loginWithPin(context, ref);
          }
        },
        initialValue: pinState.pin,
        maxLength: 4
      );
    }

    // 컴포넌트가 표시된 후 생체인식 시도
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (pinState.isBiometricsAvailable && pinState.useBiometrics) {
        _tryBiometricAuth();
      }
    });

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 40.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 60),
              Text(
                '비밀번호를 입력해주세요',
                style: AppTextStyles.bodyLarge,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 50),
              SmallPinDotsRow(currentDigit: pinState.currentDigit, onTap: handlePinInput),
              const Spacer(),
              if (pinState.isBiometricsAvailable && pinState.useBiometrics)
                Padding(
                  padding: const EdgeInsets.only(bottom: 16.0),
                  child: TextButton.icon(
                    icon: const Icon(Icons.fingerprint, color: AppColors.blueDark),
                    label: Text(
                      '생체인식으로 로그인',
                      style: AppTextStyles.bodySmall.copyWith(
                        color: AppColors.blueDark
                      ),
                    ),
                    onPressed: _tryBiometricAuth,
                  ),
                ),
            ],
          )
        )
      )
    );
  }

  // PIN 로그인 시도
  void _loginWithPin(BuildContext context, WidgetRef ref) async {
    // 로딩 표시
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Center(child: CustomLoadingIndicator()),
    );

    // 로그인 시도
    final success = await ref.read(pinStateProvider.notifier).loginWithPin();
    
    // 로딩 닫기
    Navigator.of(context).pop();

    if (success) {
      // 로그인 성공
      // 인증서와 USERKEY 있는지 확인하고 이동해야해요~~
      final secureStorage = ref.read(secureStorageProvider);
      final certificatePem = await secureStorage.read(key: StorageKeys.certificatePem);
      final userkey = await secureStorage.read(key: StorageKeys.userkey);
      
      if (context.mounted) Navigator.of(context).pop();

      // 인증서와 userkey 유무 확인
      if (certificatePem != null && userkey != null) {
        if (context.mounted) context.go('/budget');
      } else {
        if (context.mounted) context.go(SignupRoutes.getMyDataSplashPath());
      }
    } else {
      // 로그인 실패 - 에러 메시지 표시
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('로그인에 실패했습니다. PIN 번호를 확인해주세요.'),
          backgroundColor: AppColors.buttonDelete,
        ),
      );
      // PIN 초기화
      ref.read(pinStateProvider.notifier).resetPin();
    }

  }
}