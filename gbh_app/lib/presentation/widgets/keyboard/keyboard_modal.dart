// lib/presentation/widgets/keyboard/keyboard_modal.dart
import 'package:flutter/material.dart';
import 'package:marshmellow/presentation/widgets/keyboard/calculator_keyboard.dart';
import 'package:marshmellow/presentation/widgets/keyboard/numeric_keyboard.dart';
import 'package:marshmellow/presentation/widgets/keyboard/secure_numeric_keyboard.dart';

class KeyboardModal {
    // 오버레이 엔트리를 저장할 변수
  static OverlayEntry? _overlayEntry;
  
  static Future<void> showNumericKeyboard({
    required BuildContext context,
    required Function(String) onValueChanged,
    required String initialValue,
  }) async {
    await _showKeyboardModal(
      context: context,
      child: NumericKeyboard(
        onValueChanged: onValueChanged,
        initialValue: initialValue,
        onClose: () => Navigator.of(context).pop(),
      ),
    );
  }
  
  static Future<void> showCalculatorKeyboard({
    required BuildContext context,
    required Function(String) onValueChanged,
    required String initialValue,
  }) async {
    await _showKeyboardModal(
      context: context,
      child: CalculatorKeyboard(
        onValueChanged: onValueChanged,
        initialValue: initialValue,
        onClose: () => Navigator.of(context).pop(),
      ),
    );
  }
  
  static Future<void> showSecureNumericKeyboard({
    required BuildContext context,
    required Function(String) onValueChanged,
    required String initialValue,
    int maxLength = 0,
    bool obscureText = true,
  }) async {
    await _showKeyboardModal(
      context: context,
      child: SecureNumericKeyboard(
        onValueChanged: onValueChanged,
        initialValue: initialValue,
        onClose: () => Navigator.of(context).pop(),
        maxLength: maxLength,
        obscureText: obscureText,
      ),
    );
  }
  
  static Future<void> _showKeyboardModal({
    required BuildContext context,
    required Widget child,
  }) async {
    await showModalBottomSheet( // showModalBottomSheet는 배경이 되는 화면의 스크롤과 상호작용을 차단시킴 즉 스크롤 못함
      context: context,
      isScrollControlled: true,
      useSafeArea: true,        // 안전 영역 사용
      isDismissible: true,       // 외부 탭으로 닫기 가능하게 설정
      enableDrag: true,          // 드래그로 닫기 가능하게 설정
      barrierColor: Colors.transparent, // 배경색 투명하게 설정
      builder: (context) => AnimatedPadding(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(context).viewInsets.bottom,
        ),
        duration: const Duration(milliseconds: 100),
        child: Container(
          // 키보드 높이 제한
          constraints: BoxConstraints(
            maxHeight: MediaQuery.of(context).size.height * 0.8, // 화면 높이의 50%로 제한
          ),
          child: child,
        ),
      ),
    );
  }
}