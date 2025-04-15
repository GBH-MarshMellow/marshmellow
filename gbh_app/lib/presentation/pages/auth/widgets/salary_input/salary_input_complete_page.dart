import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/custom_button.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_info_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';
import 'package:intl/intl.dart';

class SalaryInputCompletePage extends ConsumerWidget {
  const SalaryInputCompletePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userInfoState = ref.watch(userInfoProvider);
    final userDetail = userInfoState.userDetail;

    // 숫자 포맷터 (천 단위 콤마)
    final currencyFormatter = NumberFormat('#,###', 'ko_KR');
    final salary = userDetail.salaryAmount ?? 0;
    final formattedSalary = currencyFormatter.format(salary);
    final salaryDay = userDetail.salaryDate != null ? '${userDetail.salaryDate}일' : '매월';

    return Scaffold(
      appBar: CustomAppbar(
        title: '',
      ),
      body: Stack(
        children: [      
          // 메인 콘텐츠
          Center(
            child: Padding(
              padding: const EdgeInsets.all(40.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Spacer(flex: 1),
                  
                  // 캐릭터 이미지
                  Image.asset(
                    'assets/images/characters/char_chair_phone.png', 
                    width: 180,
                    height: 180,
                  ),
                  
                  const SizedBox(height: 32),
                  
                  // 축하 메시지
                  Text(
                    '월급 정보 등록 완료!',
                    style: AppTextStyles.mainTitle,
                    textAlign: TextAlign.center,
                  ),
                  
                  const SizedBox(height: 16),
                  
                  // 월급 정보
                  Text(
                    '매월 $salaryDay에',
                    style: AppTextStyles.bodyLarge,
                    textAlign: TextAlign.center,
                  ),
                  
                  const SizedBox(height: 8),
                  
                  Text(
                    '$formattedSalary원이',
                    style: AppTextStyles.bodyLarge.copyWith(
                      fontSize: 24,
                      color: AppColors.backgroundBlack,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  
                  const SizedBox(height: 8),
                  
                  Text(
                    '입금될 예정이에요!',
                    style: AppTextStyles.bodyLarge,
                    textAlign: TextAlign.center,
                  ),
                  
                  const SizedBox(height: 24),
                  
                  // 추가 설명 텍스트
                  Text(
                    '월급 정보는 예산 설정과 지출 분석에 활용됩니다.',
                    style: AppTextStyles.bodySmall.copyWith(
                      color: AppColors.textSecondary,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  
                  const Spacer(flex: 1),
                  
                  // 하단 버튼
                  CustomButton(
                    text: '예산 설정하기',
                    onPressed: () {
                      // 예산 설정 페이지로 이동
                      context.go(SignupRoutes.getBudgetTypePath());
                    },
                    isEnabled: true,
                  ),
                  
                  const SizedBox(height: 16),
                  
                  // 건너뛰기 버튼
                  TextButton(
                    onPressed: () {
                      // 홈 화면으로 이동
                      context.go('/budget');
                    },
                    child: Text(
                      '나중에 할게요',
                      style: AppTextStyles.bodyMedium.copyWith(
                        color: AppColors.textSecondary,
                      ),
                    ),
                  ),
                  
                  const SizedBox(height: 40),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}