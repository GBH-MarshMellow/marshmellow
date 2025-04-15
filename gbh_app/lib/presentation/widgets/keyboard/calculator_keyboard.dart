// lib/presentation/widgets/keyboard/calculator_keyboard.dart
import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class CalculatorKeyboard extends StatefulWidget {
  final Function(String) onValueChanged;
  final String initialValue;
  final VoidCallback onClose;

  const CalculatorKeyboard({
    Key? key,
    required this.onValueChanged,
    this.initialValue = '',
    required this.onClose,
  }) : super(key: key);

  @override
  State<CalculatorKeyboard> createState() => _CalculatorKeyboardState();
}

class _CalculatorKeyboardState extends State<CalculatorKeyboard> {
  late String _currentExpression;
  late String _displayValue;

  @override
  void initState() {
    super.initState();
    _currentExpression = widget.initialValue;
    _displayValue = widget.initialValue;
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
      padding: const EdgeInsets.fromLTRB(32, 16, 32, 32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // 표시창 (옵션)
          Padding(
            padding: const EdgeInsets.only(bottom: 16.0),
            child: Container(
              width: double.infinity,
              padding:
                  const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
              decoration: BoxDecoration(
                border: Border.all(
                  color: const Color.fromARGB(255, 247, 247, 247),
                ),
                color: const Color.fromARGB(255, 247, 247, 247),
                borderRadius: BorderRadius.circular(8.0),
              ),
              child: Text(
                _displayValue,
                style: const TextStyle(fontSize: 24),
                textAlign: TextAlign.right,
              ),
            ),
          ),

          // 계산기 버튼 그리드
          GridView.count(
            crossAxisCount: 4,
            shrinkWrap: true,
            childAspectRatio: 1.3,
            mainAxisSpacing: 8.0,
            crossAxisSpacing: 0.0,
            physics: const NeverScrollableScrollPhysics(),
            children: [
              // 첫 번째 행
              _buildButton('AC', isFunction: true),
              _buildButton('+/-', isFunction: true),
              _buildButton('%', isFunction: true),
              _buildButton('/', isOperator: true),

              // 두 번째 행
              _buildButton('1'),
              _buildButton('2'),
              _buildButton('3'),
              _buildButton('×', isOperator: true),

              // 세 번째 행
              _buildButton('4'),
              _buildButton('5'),
              _buildButton('6'),
              _buildButton('+', isOperator: true),

              // 네 번째 행
              _buildButton('7'),
              _buildButton('8'),
              _buildButton('9'),
              _buildButton('-', isOperator: true),

              // 다섯 번째 행
              _buildButton('00'),
              _buildButton('0'),
              _buildButton('BACKSPACE', icon: Icons.backspace_outlined),
              _buildButton('=', isOperator: true),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildButton(String text,
      {bool isOperator = false, bool isFunction = false, IconData? icon}) {
    return ElevatedButton(
      onPressed: () => _handleKeyPress(text),
      style: ElevatedButton.styleFrom(
        shape: const CircleBorder(),
        padding: EdgeInsets.zero,
        backgroundColor: isOperator ? Colors.white : Colors.white,
        foregroundColor: isOperator
            ? Colors.black
            : isFunction
                ? Colors.black54
                : Colors.black,
        elevation: 2,
        side: BorderSide(color: AppColors.blackLight, width: 1),
      ),
      child: icon != null
          ? Icon(
              icon,
              size: 24,
              color: AppColors.blackPrimary,
            )
          : Text(text, style: AppTextStyles.appBar),
    );
  }

  void _handleKeyPress(String str) {
    switch (str) {
      case 'AC':
        setState(() {
          _currentExpression = '';
          _displayValue = '0';
        });
        break;
      case 'BACKSPACE':
        if (_currentExpression.isNotEmpty) {
          setState(() {
            _currentExpression =
                _currentExpression.substring(0, _currentExpression.length - 1);
            _displayValue =
                _currentExpression.isEmpty ? '0' : _currentExpression;
          });
        }
        break;
      case '+/-':
        setState(() {
          if (_currentExpression.startsWith('-')) {
            _currentExpression = _currentExpression.substring(1);
          } else if (_currentExpression.isNotEmpty) {
            _currentExpression = '-' + _currentExpression;
          }
          _displayValue = _currentExpression.isEmpty ? '0' : _currentExpression;
        });
        break;
      case '%':
        try {
          final value = double.parse(_currentExpression) / 100;
          setState(() {
            // 정수일 경우 소수점 제거
            _currentExpression =
                value % 1 == 0 ? value.toInt().toString() : value.toString();
            _displayValue = _currentExpression;
          });
        } catch (e) {
          // 변환 실패 시 무시
        }
        break;
      case '=':
        _calculate();
        break;
      case '+':
      case '-':
      case '×':
      case '/':
        setState(() {
          // 마지막 문자가 연산자라면 교체, 아니면 추가
          if (_currentExpression.isNotEmpty) {
            final lastChar = _currentExpression[_currentExpression.length - 1];
            if (lastChar == '+' ||
                lastChar == '-' ||
                lastChar == '×' ||
                lastChar == '/') {
              _currentExpression = _currentExpression.substring(
                      0, _currentExpression.length - 1) +
                  str;
            } else {
              _currentExpression += str;
            }
          }
          _displayValue = _currentExpression;
        });
        break;
      default:
        setState(() {
          _currentExpression += str;
          _displayValue = _currentExpression;
        });
        break;
    }

    // 소수점 확인 (.0으로 끝나는 경우 제거)
    if (_displayValue.endsWith('.0')) {
      _displayValue = _displayValue.substring(0, _displayValue.length - 2);
      _currentExpression = _displayValue;
    }

    // 상위 위젯에 값 전달
    widget.onValueChanged(_displayValue);
  }

  void _calculate() {
    try {
      // 계산식 처리
      String expression = _currentExpression;

      // 곱셈/나눗셈 기호 변환
      expression = expression.replaceAll('×', '*').replaceAll('÷', '/');

      // 계산 결과 구하기
      List<String> numbers = [];
      List<String> operators = [];

      String currentNumber = '';
      for (int i = 0; i < expression.length; i++) {
        if ('0123456789.'.contains(expression[i])) {
          currentNumber += expression[i];
        } else if ('+-*/'.contains(expression[i])) {
          if (currentNumber.isNotEmpty) {
            numbers.add(currentNumber);
            currentNumber = '';
          }
          operators.add(expression[i]);
        }
      }

      if (currentNumber.isNotEmpty) {
        numbers.add(currentNumber);
      }

      if (numbers.isEmpty) return;

      double result = double.parse(numbers[0]);
      for (int i = 0; i < operators.length; i++) {
        if (i + 1 < numbers.length) {
          double nextNum = double.parse(numbers[i + 1]);
          switch (operators[i]) {
            case '+':
              result += nextNum;
              break;
            case '-':
              result -= nextNum;
              break;
            case '*':
              result *= nextNum;
              break;
            case '/':
              if (nextNum != 0) {
                result /= nextNum;
              }
              break;
          }
        }
      }

      // 결과 표시 (소수점 처리)
      _currentExpression =
          result % 1 == 0 ? result.toInt().toString() : result.toString();
      _displayValue = _currentExpression;

      // 상위 위젯에 값 전달
      widget.onValueChanged(_displayValue);

      setState(() {});

      // 키보드 닫기
      widget.onClose();
    } catch (e) {
      // 계산 오류 처리
      setState(() {
        _displayValue = '오류';
      });
    }
  }
}
