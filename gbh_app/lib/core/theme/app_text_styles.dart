import 'package:flutter/material.dart';
import 'app_colors.dart';

// 앱 텍스트 스타일 정의
class AppTextStyles {
  // 폰트 패밀리
  static const String fontFamily = 'S-CoreDream-5';

  // 앱바
  static const TextStyle appBar = TextStyle(
    fontSize: 20,
    fontWeight: FontWeight.w500,
    color: AppColors.textPrimary,
    height: 1.0,
  );

  // 타이틀
  static const TextStyle mainTitle = TextStyle(
    fontSize: 20,
    fontWeight: FontWeight.w600,
    color: AppColors.textPrimary,
  );

  static const TextStyle subTitle = TextStyle(
    fontSize: 15,
    fontWeight: FontWeight.w600,
    color: AppColors.textPrimary,
  );

  static const TextStyle modalTitle = TextStyle(
    fontSize: 18,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary,
  );

  // 본문
  // 얇은 ExtraLarge 본문
  static const TextStyle bodyExtraLarge = TextStyle(
    fontSize: 36,
    fontWeight: FontWeight.w300,
    color: AppColors.textPrimary,
  );

  // 얇은 Large 본문
  static const TextStyle bodyLargeLight = TextStyle(
    fontSize: 20,
    fontWeight: FontWeight.w300,
    color: AppColors.textPrimary,
  );

  // Large 본문
  static const TextStyle bodyLarge = TextStyle(
    fontSize: 20,
    fontWeight: FontWeight.w500,
    color: AppColors.textPrimary,
  );

  // Medium 본문
  static const TextStyle bodyMedium = TextStyle(
    fontSize: 15,
    fontWeight: FontWeight.w500,
    color: AppColors.textPrimary,
  );

  // 얇은 Medium 본문
  static const TextStyle bodyMediumLight = TextStyle(
    fontSize: 15,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary,
  );

  // Small 본문
  static const TextStyle bodySmall = TextStyle(
    fontSize: 12,
    fontWeight: FontWeight.w500,
    color: AppColors.textPrimary,
  );

  // 얇은 ExtraSmall 본문
  static const TextStyle bodyExtraSmall = TextStyle(
    fontSize: 10,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary,
  );

  // 축하 메세지
  static const TextStyle congratulation = TextStyle(
    fontSize: 40,
    fontWeight: FontWeight.w600,
    color: AppColors.textPrimary,
  );

  // 버튼 텍스트 기본
  static const TextStyle button = TextStyle(
    fontSize: 15,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary,
  );

  // 두꺼운 버튼 텍스트
  static const TextStyle buttonBold = TextStyle(
    fontSize: 15,
    fontWeight: FontWeight.w600,
    color: AppColors.textPrimary,
  );

  // Money
  // 큰 타이틀 Money
  static const TextStyle mainMoneyTitle = TextStyle(
    fontSize: 40,
    fontWeight: FontWeight.w500,
    color: AppColors.textPrimary,
  );

  // 모달 타이틀 Money
  static const TextStyle modalMoneyTitle = TextStyle(
    fontSize: 30,
    fontWeight: FontWeight.w500,
    color: AppColors.textPrimary,
  );

  // 가계부 Money
  static const TextStyle financialLedger = TextStyle(
    fontSize: 18,
    fontWeight: FontWeight.w500,
    color: AppColors.textPrimary,
  );

  //큰 그래프 Money
  static const TextStyle moneyGraphLarge = TextStyle(
    fontSize: 18,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary,
  );

  // 중간 그래프 Money
  static const TextStyle moneyGraphMedium = TextStyle(
    fontSize: 15,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary,
  );

  static const TextStyle moneyBodyLarge = TextStyle(
    fontSize: 30,
    fontWeight: FontWeight.w200,
    color: AppColors.textPrimary,
  );

  static const TextStyle moneyBodySmall = TextStyle(
    fontSize: 8,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary,
  );

  // 기간
  static const TextStyle periodLarge = TextStyle(
    fontSize: 40,
    fontWeight: FontWeight.w500,
    color: AppColors.textPrimary,
  );

  static const TextStyle periodMedium = TextStyle(
    fontSize: 15,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary,
  );

  static const TextStyle periodSmall = TextStyle(
    fontSize: 8,
    fontWeight: FontWeight.w400,
    color: AppColors.textPrimary,
  );
}

// 사용 예시

/*
import 'package:test0316_1/core/theme/app_text_styles.dart';

// 커스텀 하는 경우 CopyWith 사용
child: Text(
                widget.errorText!,
                style: AppTextStyles.bodySmall.copyWith(
                  fontWeight: FontWeight.w300,
                  color: AppColors.warnning,
                  
                ),
              ),


// 앱바에 적용
AppBar(
  title: Text('앱 제목', style: AppTextStyles.appBar),
)

// 메인 타이틀 적용
Text('환영합니다', style: AppTextStyles.mainTitle)

// 본문 텍스트 적용
Text('앱 설명 내용입니다.', style: AppTextStyles.bodyMedium)

// 버튼에 텍스트 적용
ElevatedButton(
  child: Text('확인', style: AppTextStyles.button),
  onPressed: () {},
)

// 금액 표시
Text('₩50,000', style: AppTextStyles.mainMoneyTitle)

// 기간 표시
Text('30일', style: AppTextStyles.periodMedium)

// 모달 제목
AlertDialog(
  title: Text('알림', style: AppTextStyles.modalTitle),
  content: Text('작업이 완료되었습니다.', style: AppTextStyles.bodyMediumLight),
)

*/
