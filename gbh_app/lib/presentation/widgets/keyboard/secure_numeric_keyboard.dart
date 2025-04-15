// lib/presentation/widgets/keyboard/secure_numeric_keyboard.dart
import 'package:flutter/material.dart';
import 'dart:math';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class SecureNumericKeyboard extends StatefulWidget {
  final Function(String) onValueChanged;
  final String initialValue;
  final VoidCallback onClose;
  final int maxLength;
  final bool obscureText;
  
  const SecureNumericKeyboard({
    Key? key,
    required this.onValueChanged,
    this.initialValue = '',
    required this.onClose,
    this.maxLength = 0, // 0이면 제한 없음
    this.obscureText = true,
  }) : super(key: key);

  @override
  State<SecureNumericKeyboard> createState() => _SecureNumericKeyboardState();
}

class _SecureNumericKeyboardState extends State<SecureNumericKeyboard> {
  late String _currentValue;
  late List<String> _keypadValues;
  final List<IconData> _securityIcons = [
    Icons.fingerprint,
    Icons.face,
    Icons.shield,
    Icons.security,
    Icons.lock,
  ];
  late IconData _randomIcon;
  
  @override
  void initState() {
    super.initState();
    _currentValue = widget.initialValue;
    _randomIcon = _getRandomIcon();
    _randomizeKeypad();
  }
  
  IconData _getRandomIcon() {
    return _securityIcons[Random().nextInt(_securityIcons.length)];
  }
  
  void _randomizeKeypad() {
    // 숫자 0-9, 백스페이스, 임의 아이콘을 위한 자리 생성
    _keypadValues = [
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ICON', 'BACKSPACE'
    ];
    
    // 키패드 값 섞기
    _keypadValues.shuffle(Random());
    // 백스페이스의 현재 위치 찾기
    int backspaceIndex = _keypadValues.indexOf('BACKSPACE');
    
    // 백스페이스가 마지막 위치가 아니면 마지막 위치의 값과 교환
    if (backspaceIndex != 11) {
      // 마지막 위치의 값 임시 저장
      String lastValue = _keypadValues[11];
      
      // 백스페이스를 마지막 위치로 이동
      _keypadValues[11] = 'BACKSPACE';
      
      // 원래 마지막 위치의 값을 백스페이스의 원래 위치로 이동
      _keypadValues[backspaceIndex] = lastValue;
    }
    // 임의의 보안 아이콘 선택
    _randomIcon = _getRandomIcon();
  }
  
  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.white,
      // width: MediaQuery.of(context).size.width,
      // decoration: BoxDecoration(
      //   color: Colors.white,
      //   borderRadius: const BorderRadius.only(
      //     topLeft: Radius.circular(16.0),
      //     topRight: Radius.circular(16.0),
      //   ),
      //   border: Border.all(
      //     color: Colors.black,
      //     width: 1.0,
      //   ),
      // ),
      padding: const EdgeInsets.fromLTRB(32, 16, 32, 32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // 현재 입력값 표시 (마스킹 처리)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 20.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                for (int i = 0; i < (widget.maxLength > 0 ? widget.maxLength : 6); i++)
                Container(
                  width: 16,
                  height: 16,
                  margin: const EdgeInsets.symmetric(horizontal: 8),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: i < _currentValue.length 
                        ? AppColors.blackLight 
                        : Colors.grey.shade300,
                  ),
                ),
              ],
            ),
          ),
          
          // 보안 키패드 구현 (3x4 그리드)
          GridView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 3,
              mainAxisSpacing: 8.0,
              crossAxisSpacing: 0.0,
              childAspectRatio: 1.5,
            ),
            itemCount: 12, // 3x4 그리드
            itemBuilder: (context, index) {
              String value = _keypadValues[index];
              
              if (value == 'BACKSPACE') {
                return _buildKeypadButton(
                  'BACKSPACE', 
                  icon: Icons.backspace_outlined,
                );
              } else if (value == 'ICON') {
                return _buildKeypadButton(
                  'ICON',
                  icon: _randomIcon,
                  // isDisabled: true,
                );
              } else {
                return _buildKeypadButton(value);
              }
            },
          ),
        ],
      ),
    );
  }
    
  Widget _buildKeypadButton(String value, {IconData? icon, bool isDisabled = false}) {
    return ElevatedButton(
      onPressed: isDisabled ? null : () => _handleKeyPress(value),
      style: ElevatedButton.styleFrom(
        // shape: RoundedRectangleBorder(
        //   borderRadius: BorderRadius.circular(8),
        //   side: BorderSide(color: AppColors.blackLight, width: 1),
        // ),
        backgroundColor: Colors.white,
        foregroundColor: AppColors.blackLight,
        elevation: 0,
        padding: EdgeInsets.zero,
      ),
      child: icon != null
          ? Icon(
              icon, 
              size: 24,
              color: AppColors.blackLight,
            )
          : Text(
              value,
              style: AppTextStyles.appBar,
            ),
    );
  }
  
  void _handleKeyPress(String value) {
    switch(value) {
      case 'BACKSPACE':
        if (_currentValue.isNotEmpty) {
          setState(() {
            _currentValue = _currentValue.substring(0, _currentValue.length - 1);
            // 백스페이스 후에도 키패드 변경
            _randomizeKeypad();
          });
        }
        break;
      case 'ICON':
        // 더미 버튼 - 아무 동작 없음
        break;
      default:
        // 최대 길이 제한 확인
        if (widget.maxLength > 0 && _currentValue.length >= widget.maxLength) {
          return;
        }
        
        setState(() {
          _currentValue += value;
          // 입력할 때마다 키패드 재배열
          _randomizeKeypad();
          
          // 최대 길이 도달 시 자동 확인
          if (widget.maxLength > 0 && _currentValue.length >= widget.maxLength) {
            // 잠시 후 onClose 호출 (사용자가 마지막 입력을 볼 수 있도록)
            Future.delayed(const Duration(milliseconds: 300), () {
              widget.onClose();
            });
          }
        });
        break;
    }
    
    // 상위 위젯에 값 전달
    widget.onValueChanged(_currentValue);
  }
}