import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/di/providers/auth/certificate_process_provider.dart';
import 'package:marshmellow/di/providers/core_providers.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/certification_card.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

/*
  mm인증서 생성 성공 UI
*/
class AuthMydataCertCompletePage extends ConsumerWidget {
  const AuthMydataCertCompletePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final screenWidth = MediaQuery.of(context).size.width;
    final screenHeight = MediaQuery.of(context).size.height;
    final certProcessState = ref.watch(certificateProcessProvider);
    final secureStorage = ref.watch(secureStorageProvider);

    // 사용자 이름 가져오기
    return FutureBuilder<String?>(
      future: secureStorage.read(key: StorageKeys.userName), 
      builder: (context, snapshot) {
        // 데이터 로딩 중이면 로딩 인디케이터 표시
        if (snapshot.connectionState == ConnectionState.waiting) {
          return CustomLoadingIndicator(
            text: '정보를 불러오고 있습니다',
            backgroundColor: AppColors.whiteLight, opacity: 0.9,
          );
        }

        // 사용자 이름 가져오기
        final userName = snapshot.data ?? 'mm사용자';

        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: LayoutBuilder(
                builder: (context, constraints) {
                  return SingleChildScrollView(
                    child: ConstrainedBox(
                      constraints: BoxConstraints(
                        minHeight: constraints.maxHeight,
                      ),
                      child: IntrinsicHeight(
                        child: Column(
                          children: [
                            // 상단 여백 - 화면 크기에 비례하게 조정
                            SizedBox(height: screenHeight * 0.15),
                            
                            //인증서 카드
                            CertificateCard(
                              userName: userName, 
                              expiryDate: '2028. 03. 14.',
                              shieldSvgPath: 'assets/icons/etc/shield.svg',
                            ),
                            
                            // 중간 여백 - 화면 크기에 비례하게 조정
                            SizedBox(height: screenHeight * 0.05),
                            
                            // 텍스트 섹션
                            Text(
                              '$userName 님의 MM 인증서',
                              style: AppTextStyles.bodyLarge,
                            ),
                            const SizedBox(height: 4),
                            const Text(
                              '발급 완료!',
                              style: AppTextStyles.bodyLarge,
                            ),
                            
                            // 남은 공간을 차지하는 Spacer
                            const Spacer(),
                            
                            // 하단 여백 추가
                            const SizedBox(height: 16),
                            
                            // 버튼
                            Button(
                              text: '다음',
                              width: screenWidth * 0.9,
                              height: 56, // 약간 줄임
                              onPressed: () {
                                context.go(SignupRoutes.getMyDataSplashPath());
                              },
                            ),
                            
                            // 버튼 아래 여백
                            const SizedBox(height: 24),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
        );
      }
    );
  }
}