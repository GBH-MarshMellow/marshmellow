// lib/presentation/widgets/datepicker/date_picker_overlay.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:marshmellow/presentation/widgets/datepicker/custom_date_picker.dart';
import 'package:syncfusion_flutter_datepicker/datepicker.dart';

/// DatePicker를 오버레이로 표시하는 위젯
/// 앱의 최상위 위젯에 배치해야 함
class DatePickerOverlay extends ConsumerWidget {
  final Widget child;

  const DatePickerOverlay({Key? key, required this.child}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final datePickerState = ref.watch(datePickerProvider);

    return Stack(
      children: [
        child, // 원래 앱 내용

        // DatePicker가 표시되어야 할 때만 오버레이 표시
        if (datePickerState.isVisible && datePickerState.position != null)
          Positioned.fill(
            child: GestureDetector(
              // 배경 터치 시 DatePicker 닫기
              onTap: () =>
                  ref.read(datePickerProvider.notifier).hideDatePicker(),
              // 반투명 배경
              child: Container(
                color: Colors.black.withOpacity(0.1), // 반투명 검정 배경
              ),
            ),
          ),

        // DatePicker 컴포넌트
        if (datePickerState.isVisible && datePickerState.position != null)
          Positioned(
            left: MediaQuery.of(context).size.width * 0.05,
            top: datePickerState.position!.dy - 2,
            child: GestureDetector(
              // DatePicker 내부 터치 시 이벤트 전파 중단 (배경 터치 이벤트 방지)
              onTap: () {},
              child: Material(
                borderRadius: BorderRadius.circular(5),
                child: CustomDatePicker(
                  selectionMode: datePickerState.selectionMode,
                  initialSelectedRange: datePickerState.selectedRange,
                  initialSelectedDate: datePickerState.selectedDate,
                  initialSelectedDates: datePickerState.selectedDates,
                  onConfirm: (range) {
                    // 확인 버튼 클릭 시 선택된 범위 업데이트
                    ref
                        .read(datePickerProvider.notifier)
                        .updateSelectedRange(range);
                  },
                ),
              ),
            ),
          ),
      ],
    );
  }
}
