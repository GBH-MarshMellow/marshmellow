import 'package:flutter/material.dart';
import 'package:marshmellow/presentation/widgets/text_input/text_input.dart';

// <<<<<<<<<<<<<<<<<<<<<<< (Step 1 : 필수) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
import 'package:marshmellow/presentation/widgets/keyboard/index.dart'; // 키보드 전부 import 받아옴
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

class KeyboardTestPage extends StatefulWidget {
  const KeyboardTestPage({Key? key}) : super(key: key);

  @override
  State<KeyboardTestPage> createState() => _KeyboardTestPageState();
}

class _KeyboardTestPageState extends State<KeyboardTestPage> {

// <<<<<<<<<<<<<<<<<<<<<<< (Step 2) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  // 멤버 변수로 선언
  final TextEditingController _nameController = TextEditingController(); // 일반키보드
  final TextEditingController _numericController = TextEditingController(); // 숫자키보드용 입력값 저장
  final TextEditingController _calculatorController = TextEditingController(); // 계산기키보드용 입력값 저장

  final TextEditingController _secureController = TextEditingController(); // 보안키보드용 입력값(암호화) 저장
  String _secureValue = ''; // 보안키보드용 입력값(원본) 저장

  @override
  void dispose() {
    _nameController.dispose();
    _numericController.dispose();
    _calculatorController.dispose();
    _secureController.dispose();
    super.dispose();
  }
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('키보드 테스트 페이지'),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
        
              // <<<<<<<<<<<<<<<<<<<<<<< 숫자 키보드 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
              const Text('숫자키보드'),
              TextInput(
                label: '숫자키보드', // 라벨
                readOnly: true, 
                controller: _numericController, // 위에서 정의한 변수명 맞춰서 수정
                onTap: () async {
                  await KeyboardModal.showNumericKeyboard(
                    context: context,
                    onValueChanged: (value) {
                      setState(() {
                        _numericController.text = value;
                      });
                    },
                    initialValue: _numericController.text,
                  );
                },
              ),
              const SizedBox(height: 16),
              // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
              
              // <<<<<<<<<<<<<<<<<<<<<<< 계산기 키보드 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
              const Text('계산기키보드'),
              TextInput(
                label: '계산기키보드',
                readOnly: true, // (필수)시스템 키보드 방지
                controller: _calculatorController,
                onTap: () async {
                  await KeyboardModal.showCalculatorKeyboard(
                    context: context,
                    onValueChanged: (value) {
                      setState(() {
                        _calculatorController.text = value;
                      });
                    },
                    initialValue: _calculatorController.text,
                  );
                },
              ),
              const SizedBox(height: 16),
              // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
        
              // <<<<<<<<<<<<<<<<<<<<<<< 보안 키보드 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
              Text('보안키보드 :  $_secureValue'),
              TextInput(
                label: '보안키보드',
                readOnly: true, // (필수)시스템 키보드 방지
                controller: _secureController,
                onTap: () async {
                  // 보안 키보드를 탭할 때마다 값을 초기화
                  setState(() {
                    _secureValue = ''; // 실제 값 초기화
                    _secureController.text = ''; // 표시되는 값도 초기화
                  });
                  await KeyboardModal.showSecureNumericKeyboard(
                    context: context,
                    onValueChanged: (value) {
                      setState(() {
                        _secureValue = value;
                        _secureController.text = '•' * value.length;
                      });
                    },
                    initialValue: '',
                    maxLength: 4,
                    obscureText: true,
                  );
                },
              ),
              // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
              
              // 일반 키보드 비교 
              const Text('시스템 기본 키보드'),
              TextInput(
                label: '이름',
                controller: _nameController,
                onChanged: (value) {
                  print('입력된 이름: $value');
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}