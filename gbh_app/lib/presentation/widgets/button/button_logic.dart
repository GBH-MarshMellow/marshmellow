import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';

class ButtonLogic {
  // 버튼 상태 클래스 - 버튼의 모든 상태 정보를 캡슐화
  static ButtonState getButtonState({
    required bool isPressed,
    required bool isDisabled,
    Color? color,
    Color? disabledColor,
    Color? textColor,
    Color? borderColor,
    double? width,
    double? height,
    double? borderRadius,
    required BuildContext context,
  }) {
    final double screenWidth = MediaQuery.of(context).size.width;

    // 버튼 색상 계산
    final buttonColor = isPressed
        ? AppColors.background
        : isDisabled
            ? disabledColor ?? Colors.grey
            : color ?? AppColors.textPrimary;

    // 텍스트 색상 계산
    final buttonTextColor =
        isPressed ? AppColors.textPrimary : textColor ?? AppColors.whitePrimary;

    // 테두리 색상 계산
    final buttonBorderColor =
        isPressed ? AppColors.textPrimary : borderColor ?? Colors.transparent;

    // 크기 계산
    final buttonWidth = width ?? screenWidth * 0.9;
    final buttonHeight = height ?? 50.0;
    final buttonBorderRadius = borderRadius ?? 5.0;

    return ButtonState(
      color: buttonColor,
      textColor: buttonTextColor,
      borderColor: buttonBorderColor,
      width: buttonWidth,
      height: buttonHeight,
      borderRadius: buttonBorderRadius,
    );
  }

  // 인터랙션 여부 확인
  static bool shouldHandleInteraction(bool isDisabled) {
    return !isDisabled;
  }
}

// 버튼 상태 정보를 저장하는 클래스
class ButtonState {
  final Color color;
  final Color textColor;
  final Color borderColor;
  final double width;
  final double height;
  final double borderRadius;

  ButtonState({
    required this.color,
    required this.textColor,
    required this.borderColor,
    required this.width,
    required this.height,
    required this.borderRadius,
  });
}
