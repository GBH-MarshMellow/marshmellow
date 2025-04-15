// 핀번호 입력시 사용하는 숫자 점 표시 위젯
import 'package:flutter/widgets.dart';
import 'package:marshmellow/core/theme/app_colors.dart';

class SmallPinDotsRow extends StatelessWidget {
  final int currentDigit; // 현재 입력된 자릿수
  final Color activeColor; // 입력 된 색상
  final Color inactiveColor; // 입력 안된 색상
  final Color borderColor; // 원의 테두리 색상
  final Color inactiveborderColor; // 입력 안된 원의 테두리 색상
  final VoidCallback onTap; // 탭 햇을 때 동작

  // 고정값
  static const double _DOT_WIDTH = 60;
  static const double _DOT_HEIGHT = 90;
  static const double _DOT_SPACING = 5;
  static const double _BORDER_RADIUS = 30;
  
  const SmallPinDotsRow({
    Key? key,
    required this.currentDigit,
    required this.onTap,
    this.activeColor = AppColors.bluePrimary,
    this.inactiveColor = AppColors.whiteLight,
    this.borderColor = AppColors.blackPrimary,
    this.inactiveborderColor = AppColors.disabled,
    }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap : onTap,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: List.generate(
          4,
          (index) => Container(
            margin: const EdgeInsets.symmetric(horizontal: _DOT_SPACING),
            width: _DOT_WIDTH,
            height: _DOT_HEIGHT,
            decoration: BoxDecoration(
              color: index < currentDigit ? activeColor : inactiveColor,
              border: Border.all(color: index < currentDigit ? borderColor : inactiveborderColor),
              borderRadius: BorderRadius.circular(_BORDER_RADIUS),
            ),
          )
        ),
      )
    );
  }
}

class PinDotsRow extends StatelessWidget {
  final int currentDigit; // 현재 입력된 자릿수
  final Color activeColor; // 입력 된 색상
  final Color inactiveColor; // 입력 안된 색상
  final Color borderColor; // 원의 테두리 색상
  final Color inactiveborderColor; // 입력 안된 원의 테두리 색상
  final VoidCallback onTap; // 탭 햇을 때 동작

  // 고정값
  static const double _DOT_WIDTH = 50;
  static const double _DOT_HEIGHT = 75;
  static const double _DOT_SPACING = 2;
  static const double _BORDER_RADIUS = 30;
  
  const PinDotsRow({
    Key? key,
    required this.currentDigit,
    required this.onTap,
    this.activeColor = AppColors.bluePrimary,
    this.inactiveColor = AppColors.whiteLight,
    this.borderColor = AppColors.blackPrimary,
    this.inactiveborderColor = AppColors.disabled,
    }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap : onTap,
      child: Center(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: List.generate(
            6,
            (index) => Container(
              margin: const EdgeInsets.symmetric(horizontal: _DOT_SPACING),
              width: _DOT_WIDTH,
              height: _DOT_HEIGHT,
              decoration: BoxDecoration(
                color: index < currentDigit ? activeColor : inactiveColor,
                border: Border.all(color: index < currentDigit ? borderColor : inactiveborderColor),
                borderRadius: BorderRadius.circular(_BORDER_RADIUS),
              ),
            )
          ),
        )
      )
    );
  }
}