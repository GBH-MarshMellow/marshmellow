import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:simple_numpad/simple_numpad.dart';

class NumericKeyboard extends StatefulWidget {
  final Function(String) onValueChanged;
  final String initialValue;
  final VoidCallback onClose;
  
  const NumericKeyboard({
    Key? key,
    required this.onValueChanged,
    this.initialValue = '',
    required this.onClose,
  }) : super(key: key);

  @override
  State<NumericKeyboard> createState() => _NumericKeyboardState();
}

class _NumericKeyboardState extends State<NumericKeyboard> {
  late String _currentValue;
  
  @override
  void initState() {
    super.initState();
    _currentValue = widget.initialValue;
  }
  
  @override
  Widget build(BuildContext context) {
    return Container(
      width: MediaQuery.of(context).size.width,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(16.0),
          topRight: Radius.circular(16.0),
        ),
        border: Border.all(
          color: Colors.black,
          width: 1.0,
        ),
      ),
      padding: const EdgeInsets.fromLTRB(18.0, 16.0, 18.0, 28.0),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // 상단 바 (드래그 핸들)
          Container(
            width: 40,
            height: 4,
            margin: const EdgeInsets.only(bottom: 16.0),
            decoration: BoxDecoration(
              color: Colors.grey[300],
              borderRadius: BorderRadius.circular(2.0),
            ),
          ),
          
          // 표시창
          Container(
            width: 240,
            margin: const EdgeInsets.only(bottom: 16.0),
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
            decoration: BoxDecoration(
              border: Border.all(color: const Color.fromARGB(255, 247, 247, 247)),
              color: const Color.fromARGB(255, 247, 247, 247),
              borderRadius: BorderRadius.circular(8.0),
            ),
            child: Text(
              _currentValue,
              style: const TextStyle(fontSize: 24),
              textAlign: TextAlign.right,
            ),
          ),
          
          // 키패드
          SimpleNumpad(
            buttonWidth: 60,
            buttonHeight: 60,
            gridSpacing: 20,
            buttonBorderRadius: 30,
            foregroundColor: Colors.red,
            buttonBorderSide: const BorderSide(
              color: AppColors.blackLight,
              width: 1,
            ),
            textStyle: AppTextStyles.appBar,
            useBackspace: true,
            optionText: 'clear',
            onPressed: (str) {
              _handleKeyPress(str);
            },
          ),
        ],
      ),
    );
  }
  
  void _handleKeyPress(String str) {
    switch(str) {
      case 'BACKSPACE':
        if (_currentValue.isNotEmpty) {
          setState(() {
            _currentValue = _currentValue.substring(0, _currentValue.length - 1);
          });
        }
        break;
      case 'clear':
        setState(() {
          _currentValue = '';
        });
        break;
      default:
        setState(() {
          _currentValue += str;
        });
        break;
    }
    
    // 상위 위젯에 값 전달
    widget.onValueChanged(_currentValue);
  }
}