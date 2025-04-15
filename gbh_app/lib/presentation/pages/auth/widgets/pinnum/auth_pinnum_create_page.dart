import 'package:go_router/go_router.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/src/widgets/framework.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/di/providers/auth/user_provider.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/di/providers/auth/pin_provider.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/pinnum/biometric_option.dart';
import 'package:marshmellow/presentation/widgets/dots_input/dots_input.dart';
import 'package:marshmellow/presentation/widgets/keyboard/index.dart';

/*
  핀넘버 설정 시작 UI
*/
class AuthPinnumPage extends ConsumerWidget {
    const AuthPinnumPage({Key? key}) : super(key: key);

    @override
    Widget build(BuildContext context, WidgetRef ref) {
        final pinState = ref.watch(pinStateProvider);
        final userState = ref.watch(userStateProvider);

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

              // PIN 4자리 되면 자동으로 다음 단계로 이동
              if (value.length == 4) {
                if (!pinState.isConfirmingPin) {
                  // 첫 입력 후 확인 모드로 전환
                  final tempPin = value;
                  Future.delayed(const Duration(milliseconds: 300), () {
                    ref.read(previousPinProvider.notifier).state = tempPin;
                    notifier.setConfirmMode(true);
                  });
                } else {
                  // 확인 모드에서 PIN 저장 시도
                  final previousPin = ref.read(previousPinProvider);
                  
                  // 핀 번호가 일치하는지 확인
                  if (value != previousPin) {
                    // 일치하지 않을 경우 에러 메시지 표시
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('PIN 번호가 일치하지 않습니다. 다시 시도해주세요.'),
                        backgroundColor: AppColors.buttonDelete,
                      ),
                    );
                    // 초기화 후 다시 입력받기
                    Future.delayed(const Duration(milliseconds: 1000), () {
                      notifier.resetPin();
                      notifier.setConfirmMode(false);
                    });
                    return;
                  }
              
                  // PIN 저장 시도
                  Future.delayed(const Duration(milliseconds: 300), () async {
                    // 화면에 로딩 표시
                    showDialog(
                      context: context,
                      barrierDismissible: false,
                      builder: (context) => const Center(child: CircularProgressIndicator()),
                    );
                    
                    final success = await notifier.savePin(previousPin);
                    
                    // 로딩 dialog 닫기
                    Navigator.of(context).pop();
                    
                    if (success) {
                      // 성공 시 완료 페이지로 이동
                      context.go(SignupRoutes.getPinCompletePath());
                    } else {
                      // 실패 시 에러 메시지 표시
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                          content: Text('PIN 번호 설정에 실패했습니다. 다시 시도해주세요.'),
                          backgroundColor: AppColors.buttonDelete,
                        ),
                      );
                      notifier.resetPin();
                      notifier.setConfirmMode(false);
                    }
                  });
                }
              }
            },
            initialValue: pinState.pin,
            maxLength: 4
          );
        }

        return Scaffold(
            body: SafeArea(
                child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 40.0),
                    child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          const SizedBox(height: 60),
                          Text(
                            pinState.isConfirmingPin
                              ? '비밀번호를 \n한 번 더 눌러주세요'
                              : 'MM을 안전하게 쓰려면 \n 비밀번호가 필요해요',
                            style: AppTextStyles.bodyLarge,
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 50,),
                          SmallPinDotsRow(currentDigit: pinState.currentDigit, onTap: handlePinInput,),
                          const Spacer(),
                          if (pinState.isBiometricsAvailable)
                            BiometricOption(),
                        ],
                    )
                )
            )
        );
    }
}

class BiometricOption extends ConsumerWidget {
  const BiometricOption({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final pinState = ref.watch(pinStateProvider);

    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: SwitchListTile(
        title: Row(
          children: [
            const Icon(Icons.fingerprint, color: AppColors.greyPrimary),
            const SizedBox(width: 8),
            Text(
              '다음부터 생체인식 사용하기',
              style: AppTextStyles.bodyExtraSmall.copyWith(
                color: AppColors.blackLight
              ),
            ),
          ],
        ),
        value: pinState.useBiometrics, 
        onChanged: (value) {
          ref.read(pinStateProvider.notifier).toggleBiometrics(value);
        },
        activeColor: AppColors.blueDark,
        contentPadding: EdgeInsets.zero,
      )
    );
  }
}