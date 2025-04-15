// lib/presentation/widgets/loading/custom_loading_indicator.dart
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class CustomLoadingIndicator extends StatelessWidget {
  final String text; // 메세지 내용
  final double opacity; // 투명도
  final Color backgroundColor; // 배경색
  
  const CustomLoadingIndicator({
    Key? key, 
    this.text = "이곳에 text 입력",
    this.opacity = 0.7,
    this.backgroundColor = Colors.black, // 기본값은 검정색
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      color: backgroundColor.withOpacity(opacity), // 배경색과 불투명도 함께 적용
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // 애벌레 Lottie 애니메이션
            Container(
              width: 150,
              height: 150,
              margin: const EdgeInsets.only(left: 40), // 중앙 정렬하려고 하드코딩했습니다
              child: Lottie.asset(
              'assets/images/loading/temp2_worm.json',
              fit: BoxFit.contain,
              ),
            ),

            // 텍스트 부분
            Text(
              text,
              style: AppTextStyles.mainTitle,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}