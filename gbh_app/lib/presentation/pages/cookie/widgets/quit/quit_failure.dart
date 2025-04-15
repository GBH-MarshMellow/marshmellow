import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/constants/icon_path.dart';

class QuitFailure extends StatelessWidget {
  const QuitFailure({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        // 이미지와 첫 번째 메시지는 Stack으로 오버레이
        SizedBox(
          height: 380, // 이미지 높이에 맞게 조정
          child: Stack(
            alignment: Alignment.center,
            children: [
              // 경고 이미지 (카드5 사용)
              Image.asset(
                IconPath.card5,
                width: 300,
                height: 380,
                fit: BoxFit.contain,
              ),

              // 오버레이 텍스트 (첫 번째 경고 메시지만)
              Positioned(
                top: 60, // 상단 여백 조정
                left: 0,
                right: 0,
                child: Text(
                  '지금 퇴사하면\n큰일 나요!',
                  textAlign: TextAlign.center,
                  style: AppTextStyles.bodyLarge.copyWith(
                    color: AppColors.textPrimary,
                    fontWeight: FontWeight.bold,
                    fontSize: 22,
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 15),
        Text(
          '  3개월 내에 자금이 바닥날 예정입니다.',
          textAlign: TextAlign.center,
          style: AppTextStyles.bodyLarge.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 15),
        Text(
          '상단의 "?" 버튼을 눌러 \n 퇴사 꿀팁을 읽고 준비해보세요!',
          textAlign: TextAlign.center,
          style:
              AppTextStyles.bodyMedium.copyWith(color: AppColors.textSecondary),
        ),
      ],
    );
  }
}
