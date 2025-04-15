// lib/presentation/widgets/date_picker_button.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:syncfusion_flutter_datepicker/datepicker.dart';

/// DatePicker를 쉽게 사용할 수 있는 버튼 위젯
class DatePickerButton extends ConsumerWidget {
  final Widget child; // 버튼 내용 (주로 Text 위젯)
  final ButtonStyle? style; // 버튼 스타일
  final DateRangePickerSelectionMode selectionMode; // 선택 모드
  final Function(PickerDateRange)? onDateRangeSelected; // 날짜 범위 선택 완료 콜백

  const DatePickerButton({
    Key? key,
    required this.child,
    this.style,
    this.selectionMode = DateRangePickerSelectionMode.range,
    this.onDateRangeSelected,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // 현재 선택된 날짜 범위 구독
    final datePickerState = ref.watch(datePickerProvider);
    
    return ElevatedButton(
      style: style,
      onPressed: () {
        // 버튼의 위치 정보 가져오기
        final RenderBox renderBox = context.findRenderObject() as RenderBox;
        final position = renderBox.localToGlobal(Offset.zero);
        final size = renderBox.size;
        
        // DatePicker 표시 위치 계산 (버튼 중앙 하단)
        final Offset pickerPosition = Offset(
          position.dx + (size.width / 2),
          position.dy + size.height,
        );
        
        // DatePicker 표시
        ref.read(datePickerProvider.notifier).showDatePicker(
          position: pickerPosition,
          selectionMode: selectionMode,
          initialRange: datePickerState.selectedRange,
        );
      },
      child: child,
    );
  }
}