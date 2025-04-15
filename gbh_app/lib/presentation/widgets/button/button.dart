import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/button/button_logic.dart';

class Button extends StatefulWidget {
  final String text;
  final VoidCallback? onPressed;
  final double? width;
  final double? height;
  final Color? color;
  final Color? disabledColor;
  final Color? borderColor;
  final Color? textColor;
  final bool isDisabled;
  final bool isLoading; // 추가된 매개변수
  final Color? loadingColor; // 로딩 인디케이터 색상
  final TextStyle? textStyle;
  final double? borderRadius;

  const Button({
    super.key,
    this.text = '저장하기',
    this.onPressed,
    this.width,
    this.height,
    this.color = AppColors.textPrimary,
    this.disabledColor = Colors.grey,
    this.borderColor,
    this.textColor = AppColors.whitePrimary,
    this.isDisabled = false,
    this.isLoading = false, // 기본값 설정
    this.loadingColor, // 로딩 인디케이터 색상 (기본값은 textColor와 동일)
    this.textStyle,
    this.borderRadius = 5,
  });

  @override
  State<Button> createState() => _ButtonState();
}

class _ButtonState extends State<Button> {
  bool _isPressed = false;

  void _handleHighlightChanged(bool isHighlighted) {
    if (ButtonLogic.shouldHandleInteraction(widget.isDisabled || widget.isLoading)) {
      setState(() {
        _isPressed = isHighlighted;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    // 로딩 중이거나 비활성화 상태인 경우
    final bool isEffectivelyDisabled = widget.isDisabled || widget.isLoading;

    // Button Logic에서 가져오기기
    final buttonState = ButtonLogic.getButtonState(
      isPressed: _isPressed,
      isDisabled: isEffectivelyDisabled,
      color: widget.color,
      disabledColor: widget.disabledColor,
      textColor: widget.textColor,
      borderColor: widget.borderColor,
      width: widget.width,
      height: widget.height,
      borderRadius: widget.borderRadius,
      context: context,
    );

    // 텍스트 스타일 적용
    final effectiveTextStyle = widget.textStyle ??
        AppTextStyles.bodyMedium.copyWith(
          color: buttonState.textColor,
        );
    
    // 로딩 인디케이터 색상 (textColor와 같게 설정)
    final loadingColor = widget.loadingColor ?? buttonState.textColor;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        splashColor: Colors.transparent,
        highlightColor: Colors.transparent,
        onTap: isEffectivelyDisabled ? null : widget.onPressed,
        onHighlightChanged: isEffectivelyDisabled ? null : _handleHighlightChanged,
        borderRadius: BorderRadius.circular(buttonState.borderRadius),
        child: Ink(
          width: buttonState.width,
          height: buttonState.height,
          decoration: BoxDecoration(
            color: buttonState.color,
            border: Border.all(color: buttonState.borderColor),
            borderRadius: BorderRadius.circular(buttonState.borderRadius),
          ),
          child: Center(
            child: widget.isLoading
                // 로딩 중인 경우 로딩 인디케이터 표시
                ? SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(loadingColor),
                    ),
                  )
                // 로딩 중이 아닌 경우 텍스트 표시
                : Text(
                    widget.text,
                    style: effectiveTextStyle,
                  ),
          ),
        ),
      ),
    );
  }
}

// ===================== 사용 예시 =====================
/*
// Button 위젯 사용 예시

// 1. 기본 사용법
Button(
  text: '로그인',
  onPressed: () {
    print('로그인 버튼 클릭');
    // 로그인 처리 로직
  },
)

// 2. 투명 배경, 테두리 있는 버튼
Button(
  text: '가입하기',
  color: Colors.transparent,
  borderColor: AppColors.textPrimary,
  textColor: AppColors.textPrimary,
  onPressed: () {
    // 가입 처리 로직
  },
)

// 3. 둥근 테두리 버튼
Button(
  text: '다음',
  color: Colors.transparent,
  borderColor: AppColors.textPrimary,
  textColor: AppColors.textPrimary,
  borderRadius: 30,
  onPressed: () {
    // 다음 페이지로 이동
  },
)

// 4. 버튼 크기 조정
Button(
  text: '취소',
  width: 200, // 직접 너비 지정
  height: 45, // 직접 높이 지정
  onPressed: () {
    // 취소 처리 로직
  },
)

// 5. 비활성화된 버튼
Button(
  text: '제출',
  isDisabled: true,
  disabledColor: Colors.grey.shade300,
  onPressed: () {
    // 비활성화된 경우 호출되지 않음
  },
)

// 6. 사용자 지정 텍스트 스타일
Button(
  text: '확인',
  textStyle: TextStyle(
    fontSize: 18,
    fontWeight: FontWeight.bold,
    color: Colors.white,
  ),
  onPressed: () {
    // 확인 처리 로직
  },
)

// 7. 로딩 상태 버튼
Button(
  text: '저장',
  isLoading: true, // 로딩 중 상태 설정
  onPressed: () {
    // 로딩 중일 때는 호출되지 않음
  },
)

// 8. 로딩 상태 버튼 (커스텀 로딩 색상)
Button(
  text: '로그인',
  isLoading: isAuthenticating, // 상태 변수에 따라 로딩 표시
  loadingColor: Colors.white, // 로딩 인디케이터 색상 설정
  color: AppColors.bluePrimary,
  onPressed: () => _login(),
)

// 9. 두 개의 버튼을 가로로 나란히 배치
Row(
  mainAxisAlignment: MainAxisAlignment.center,
  children: [
    Button(
      text: '네',
      width: MediaQuery.of(context).size.width * 0.43, // 화면 너비의 43%
      borderRadius: 30,
      onPressed: () {
        print('네 선택됨');
      },
    ),
    const SizedBox(width: 10), // 버튼 사이 간격
    Button(
      text: '아니오',
      width: MediaQuery.of(context).size.width * 0.43, // 화면 너비의 43%
      borderRadius: 30,
      color: Colors.transparent,
      borderColor: AppColors.textPrimary,
      textColor: AppColors.textPrimary,
      onPressed: () {
        print('아니오 선택됨');
      },
    ),
  ],
)
*/