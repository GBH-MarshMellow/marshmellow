// lib/presentation/pages/test/datepickertest.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/presentation/widgets/text_input/text_input.dart';
import 'package:marshmellow/presentation/widgets/datepicker/date_picker_button.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:syncfusion_flutter_datepicker/datepicker.dart';

// StatefulWidget으로 변경
class Datepickertest extends ConsumerStatefulWidget {
  const Datepickertest({super.key});

  @override
  ConsumerState<Datepickertest> createState() => _DatepickertestState();
}

class _DatepickertestState extends ConsumerState<Datepickertest> {
  // TextEditingController 추가
  final TextEditingController _nameController = TextEditingController();

  // dispose 메서드 추가
  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  // 날짜 형식을 일관성 있게 표시하는 함수
  String _formatDate(DateTime date) {
    return '${date.year}.${date.month.toString().padLeft(2, '0')}.${date.day.toString().padLeft(2, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    // 현재 선택된 날짜 범위 가져오기
    final datePickerState = ref.watch(datePickerProvider);
    String selectedRange = "날짜를 선택하세요";
    
    // 선택된 날짜 범위가 있으면 표시
    if (datePickerState.selectedRange != null) {
      final startDate = datePickerState.selectedRange!.startDate;
      final endDate = datePickerState.selectedRange!.endDate;
      
      if (startDate != null) {
        selectedRange = _formatDate(startDate);
        
        if (endDate != null && endDate != startDate) {
          selectedRange += " - ${_formatDate(endDate)}";
        }
      }
    }
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('테스트 페이지'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: Container(
          color: Colors.green,
          width: MediaQuery.of(context).size.width*0.9, // 화면의 90%
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,  // 시작 부분부터 배치
            crossAxisAlignment: CrossAxisAlignment.start,  // 좌측 정렬 추가
            children: [
              const Text(
                '테스트 페이지입니다',
                style: TextStyle(fontSize: 24),
              ),
              const SizedBox(height: 20),
              
              // 선택된 날짜 범위 표시
              Text(selectedRange, style: TextStyle(fontSize: 16)),
              
              const SizedBox(height: 20),
              
              // DatePicker 버튼
              DatePickerButton(
                child: const Text("전체"),
                selectionMode: DateRangePickerSelectionMode.range,
                // style: ButtonStyle(backgroundColor: MaterialStateProperty.all(Colors.blue)), // 스타일적용방법 1
                style: ElevatedButton.styleFrom( backgroundColor: Colors.blue,),// 스타일적용방법 2
                onDateRangeSelected: (range) {
                  if (range != null) {
                    // 개발용 로그
                    final startDate = range.startDate;
                    final endDate = range.endDate;
                    
                    String logMessage = "선택된 날짜 범위: ";
                    if (startDate != null) {
                      logMessage += _formatDate(startDate);
                      if (endDate != null) {
                        logMessage += " ~ ${_formatDate(endDate)}";
                      }
                    }
                    
                    debugPrint(logMessage);
                  }
                },
              ),
              
              const SizedBox(height: 20),
              
              ElevatedButton(
                onPressed: () {
                  context.pop();
                },
                child: const Text('홈으로 돌아가기'),
              ),

              // TextInput 위젯 추가
              TextInput(
                label: '이름',
                controller: _nameController,
                onChanged: (value) {
                  debugPrint('입력된 이름: $value');
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}