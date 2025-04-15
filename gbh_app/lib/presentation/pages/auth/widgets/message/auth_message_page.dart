import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/di/providers/auth/identity_verification_provider.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/custom_button.dart';

/*
  본인인증 메세지 전송 UI
*/
class AuthMessagePage extends ConsumerStatefulWidget {
  final Map<String, dynamic> userInfo;

  const AuthMessagePage({
    Key? key,
    required this.userInfo
  }) : super(key: key);

  @override
  ConsumerState<AuthMessagePage> createState() => _AuthMessagePageState();
}

class _AuthMessagePageState extends ConsumerState<AuthMessagePage> {
  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    // 필요시 리소스 정리
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 인증 상태 감시
    final verificationState = ref.watch(identityVerificationProvider);
    // 인증 상태에 따라 버튼 활성화
    final isButtonEnabled = verificationState.status == VerificationStatus.emailSent ||
                            verificationState.status == VerificationStatus.connectionClosed;
    // 인증 코드 만료 또는 실패 상태 확인(재요청버튼으로전환하기위함)
    final isCodeExpiredOrFailed = verificationState.status == VerificationStatus.expired ||
                                  verificationState.status == VerificationStatus.failed;

    // 인증 상태에 따라 UI 업데이트
    if (verificationState.status == VerificationStatus.verified) {
      // 인증 완료시 다음 단계로 자동 이동
      WidgetsBinding.instance.addPostFrameCallback((_) {
        context.go(SignupRoutes.getAuthCompletePath());
      });
    }

    final screenHeight = MediaQuery.of(context).size.height;

    final serverEmail = widget.userInfo['serverEmail'] ?? '';
    final verificationCode = widget.userInfo['verificationCode'] ?? '';
    final expiresIn = widget.userInfo['expiresIn'];

    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // 상단 설명 텍스트
            const SizedBox(height: 130),
            const Text('본인인증을 위해', style: AppTextStyles.mainTitle),
            const Text('문자를 보내주세요', style: AppTextStyles.mainTitle),
            const SizedBox(height: 10),
            Text('내용은 mm이 써두었으니', style: AppTextStyles.bodySmall.copyWith(color: AppColors.disabled)),
            Text('[문자 보내기]만 눌러주세요', style: AppTextStyles.bodySmall.copyWith(color: AppColors.disabled)),
            const SizedBox(height: 10),
            if (widget.userInfo['expiredIn'] != null)
              Text(
                '인증코드 유효시간: ${widget.userInfo['expiredIn']}초', 
                style: AppTextStyles.bodySmall.copyWith(color: AppColors.disabled)),

            // 인증 상태 메시지 표시
            if (verificationState.status == VerificationStatus.verifying)
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Text(
                  '인증 확인 정보를 불러오는 중입니다...',
                  style: AppTextStyles.bodySmall.copyWith(color: AppColors.blueDark),
                ),
              ),

            if (verificationState.status == VerificationStatus.expired)
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Text(
                  '인증 코드가 만료되었습니다. 다시 시도해주세요.',
                  style: AppTextStyles.bodySmall.copyWith(color: Colors.red),
                ),
              ),

            if (verificationState.status == VerificationStatus.failed && 
                verificationState.errorMessage != null)
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Text(
                  verificationState.errorMessage!,
                  style: AppTextStyles.bodySmall.copyWith(color: Colors.red),
                ),
              ),

            const Spacer(),

            // 이미지
            Image.asset(
              'assets/images/userverification.png',
              fit:BoxFit.contain,
            ),

            // 문자 보내기 버튼
            Padding(
              padding: const EdgeInsets.only(top: 0),
              child: isCodeExpiredOrFailed
                ? CustomButton(
                  text: '인증 코드 재요청', 
                  onPressed: () => _requestNewCode(context),
                  isEnabled: isButtonEnabled,
                  )
                : CustomButton(
                  text: '문자 보내기', 
                  onPressed: () => _sendAuthMessage(context),
                  isEnabled: isButtonEnabled,
                  ),
            ),
            SizedBox(height: screenHeight * 0.05),
          ],
        ),
      ),
    );
  }

  void _sendAuthMessage(BuildContext context) async {

    final serverEmail = widget.userInfo['serverEmail'] ?? '';
    final verificationCode = widget.userInfo['verificationCode'] ?? '';
    final messageBody = '$verificationCode';

    // SMS 앱 열기
    final uri = Uri(
      scheme: 'sms',
      path: serverEmail, // 문자 보낼 사람
      queryParameters: {'body': messageBody},
    );

    try {
      if (await canLaunchUrl(uri)) {
        await launchUrl(uri);

        // 인증중 로딩 페이지로 이동
        context.go(SignupRoutes.getAuthLoadingPath());
      } else {
        // SMS 앱 못 열 경우 클립보드에 복사 후 안내하기
        await Clipboard.setData(ClipboardData(text: messageBody));
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('메시지 앱을 열 수 없어 메시지를 클립보드에 복사했습니다.'),
            duration: const Duration(seconds: 3),
            action: SnackBarAction(
              label: '확인', 
              onPressed: () {
                // 인증중 로딩 페이지로 이동
                context.go(SignupRoutes.getAuthLoadingPath());
              },
            ),
          )
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('메시지 앱을 열 수 없습니다: $e'),)
      );
    }
  }

  // 인증 코드 재요청
  void _requestNewCode(BuildContext context) {
    // 전화번호 가져오기
    final phone = widget.userInfo['phone'] as String;

    // 인증 요청 다시 시작
    ref.read(identityVerificationProvider.notifier).verifyIdentity(phone);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('인증 코드를 재요청했습니다. 잠시만 기다려주세요.'))
    );
  }
}

