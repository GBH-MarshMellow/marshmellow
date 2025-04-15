import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class DateTimeWheelPicker extends ConsumerStatefulWidget {
  final DateTime initialDateTime;
  final DateTime? minimumDate;
  final DateTime? maximumDate;
  final Function(DateTime) onDateTimeChanged;
  final String confirmButtonText;
  final String cancelButtonText;
  final String nextButtonText;
  final CupertinoDatePickerMode initialMode;

  const DateTimeWheelPicker({
    super.key,
    required this.initialDateTime,
    this.minimumDate,
    this.maximumDate,
    required this.onDateTimeChanged,
    this.confirmButtonText = '확인',
    this.cancelButtonText = '취소',
    this.nextButtonText = '다음',
    this.initialMode = CupertinoDatePickerMode.date,
  });

  @override
  ConsumerState<DateTimeWheelPicker> createState() =>
      _DateTimeWheelPickerState();
}

class _DateTimeWheelPickerState extends ConsumerState<DateTimeWheelPicker> {
  late DateTime _selectedDateTime;
  late CupertinoDatePickerMode _currentMode;

  // 페이지 컨트롤러를 사용하여 날짜/시간 선택 화면 전환
  late PageController _pageController;

  @override
  void initState() {
    super.initState();
    // 초기 날짜 설정 시 최소/최대 날짜 범위 내에 있도록 조정
    _selectedDateTime = _adjustInitialDateTime();
    _currentMode = widget.initialMode;
    _pageController = PageController(
        initialPage: _currentMode == CupertinoDatePickerMode.date ? 0 : 1);
  }

  // 초기 날짜를 최소/최대 범위 내로 조정하는 메서드
  DateTime _adjustInitialDateTime() {
    DateTime result = widget.initialDateTime;

    // 최소 날짜가 설정되어 있고, 초기 날짜가 최소 날짜보다 이전인 경우
    if (widget.minimumDate != null && result.isBefore(widget.minimumDate!)) {
      // 날짜만 최소 날짜로 설정하고 시간은 유지
      result = DateTime(
        widget.minimumDate!.year,
        widget.minimumDate!.month,
        widget.minimumDate!.day,
        result.hour,
        result.minute,
      );
    }

    // 최대 날짜가 설정되어 있는 경우
    if (widget.maximumDate != null) {
      // 최대 날짜의 시간을 23:59:59로 설정하여 당일 전체를 포함하도록 함
      final adjustedMaxDate = DateTime(
        widget.maximumDate!.year,
        widget.maximumDate!.month,
        widget.maximumDate!.day,
        23,
        59,
        59,
      );

      // 초기 날짜가 조정된 최대 날짜를 초과하는 경우
      if (result.isAfter(adjustedMaxDate)) {
        // 날짜는 최대 날짜로 하되 시간은 유지 (단, 최대 23:59까지만)
        result = DateTime(
          widget.maximumDate!.year,
          widget.maximumDate!.month,
          widget.maximumDate!.day,
          result.hour > 23 ? 23 : result.hour,
          result.minute > 59 ? 59 : result.minute,
        );
      }
    }

    return result;
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  // 날짜 포맷 - 날짜만 표시
  String _formatDate(DateTime date) {
    return '${date.year}년 ${date.month}월 ${date.day}일';
  }

  // 시간 포맷
  String _formatTime(DateTime date) {
    return '${date.hour.toString().padLeft(2, '0')}:${date.minute.toString().padLeft(2, '0')}';
  }

  // 날짜 선택 모드로 전환
  void _goToDateMode() {
    setState(() {
      _currentMode = CupertinoDatePickerMode.date;
    });
    _pageController.animateToPage(
      0,
      duration: Duration(milliseconds: 300),
      curve: Curves.easeInOut,
    );
  }

  // 시간 선택 모드로 전환
  void _goToTimeMode() {
    setState(() {
      _currentMode = CupertinoDatePickerMode.time;
    });
    _pageController.animateToPage(
      1,
      duration: Duration(milliseconds: 300),
      curve: Curves.easeInOut,
    );
  }

  @override
  Widget build(BuildContext context) {
    // 날짜 선택용 최대/최소 날짜 - 2000년부터 2025년까지만 선택 가능하게 설정
    final datePickerMaxDate = DateTime(2025, 12, 31, 23, 59, 59);
    final datePickerMinDate = DateTime(2000, 1, 1);

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const SizedBox(height: 30),
        // PageView를 사용하여 날짜/시간 선택기 전환
        SizedBox(
          height: 200,
          child: PageView(
            controller: _pageController,
            physics: NeverScrollableScrollPhysics(), // 스와이프 비활성화
            onPageChanged: (index) {
              setState(() {
                _currentMode = index == 0
                    ? CupertinoDatePickerMode.date
                    : CupertinoDatePickerMode.time;
              });
            },
            children: [
              // 날짜 선택 페이지
              CupertinoDatePicker(
                key: ValueKey('date-picker'),
                mode: CupertinoDatePickerMode.date,
                initialDateTime: _selectedDateTime,
                minimumDate: datePickerMinDate,
                maximumDate: datePickerMaxDate,
                dateOrder: DatePickerDateOrder.ymd,
                use24hFormat: true,
                onDateTimeChanged: (DateTime newDate) {
                  setState(() {
                    _selectedDateTime = DateTime(
                      newDate.year,
                      newDate.month,
                      newDate.day,
                      _selectedDateTime.hour,
                      _selectedDateTime.minute,
                    );
                  });
                },
              ),

              // 시간 선택 페이지
              CupertinoDatePicker(
                key: ValueKey('time-picker'),
                mode: CupertinoDatePickerMode.time,
                initialDateTime: _selectedDateTime,
                use24hFormat: true,
                onDateTimeChanged: (DateTime newDate) {
                  setState(() {
                    // 시간이 변경되었을 때 최대/최소 날짜 제약 조건을 확인
                    DateTime updatedDateTime = DateTime(
                      _selectedDateTime.year,
                      _selectedDateTime.month,
                      _selectedDateTime.day,
                      newDate.hour,
                      newDate.minute,
                    );

                    // 최대 날짜 제약 조건 검사
                    if (widget.maximumDate != null &&
                        _selectedDateTime.year == widget.maximumDate!.year &&
                        _selectedDateTime.month == widget.maximumDate!.month &&
                        _selectedDateTime.day == widget.maximumDate!.day) {
                      // 당일이 최대 날짜인 경우만 시간 제약을 적용
                      final maxTimeOfDay = TimeOfDay(hour: 23, minute: 59);
                      final selectedTimeOfDay =
                          TimeOfDay(hour: newDate.hour, minute: newDate.minute);

                      // 선택된 시간이 최대 시간을 초과하는 경우 최대 시간으로 제한
                      if ((selectedTimeOfDay.hour > maxTimeOfDay.hour) ||
                          (selectedTimeOfDay.hour == maxTimeOfDay.hour &&
                              selectedTimeOfDay.minute > maxTimeOfDay.minute)) {
                        updatedDateTime = DateTime(
                          _selectedDateTime.year,
                          _selectedDateTime.month,
                          _selectedDateTime.day,
                          maxTimeOfDay.hour,
                          maxTimeOfDay.minute,
                        );
                      }
                    }

                    // 최소 날짜 제약 조건 검사
                    if (widget.minimumDate != null &&
                        _selectedDateTime.year == widget.minimumDate!.year &&
                        _selectedDateTime.month == widget.minimumDate!.month &&
                        _selectedDateTime.day == widget.minimumDate!.day) {
                      // 당일이 최소 날짜인 경우만 시간 제약을 적용
                      final minTimeOfDay = TimeOfDay(
                          hour: widget.minimumDate!.hour,
                          minute: widget.minimumDate!.minute);
                      final selectedTimeOfDay =
                          TimeOfDay(hour: newDate.hour, minute: newDate.minute);

                      // 선택된 시간이 최소 시간보다 이전인 경우 최소 시간으로 제한
                      if ((selectedTimeOfDay.hour < minTimeOfDay.hour) ||
                          (selectedTimeOfDay.hour == minTimeOfDay.hour &&
                              selectedTimeOfDay.minute < minTimeOfDay.minute)) {
                        updatedDateTime = DateTime(
                          _selectedDateTime.year,
                          _selectedDateTime.month,
                          _selectedDateTime.day,
                          minTimeOfDay.hour,
                          minTimeOfDay.minute,
                        );
                      }
                    }

                    _selectedDateTime = updatedDateTime;
                  });
                },
              ),
            ],
          ),
        ),

        // 버튼 영역
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              TextButton(
                onPressed: () {
                  if (_currentMode == CupertinoDatePickerMode.time) {
                    // 시간 선택 모드에서 이전 버튼 누르면 날짜 선택 모드로 돌아감
                    _goToDateMode();
                  } else {
                    // 날짜 선택 모드에서 취소 버튼 누르면 모달 닫힘
                    Navigator.of(context).pop();
                  }
                },
                child: Text(
                  _currentMode == CupertinoDatePickerMode.time
                      ? '이전'
                      : widget.cancelButtonText,
                  style: AppTextStyles.bodyMedium.copyWith(
                    color: AppColors.textSecondary,
                  ),
                ),
              ),
              TextButton(
                onPressed: () {
                  if (_currentMode == CupertinoDatePickerMode.date) {
                    // 날짜 선택 모드에서 다음 버튼 누르면 시간 선택 모드로 전환
                    _goToTimeMode();
                  } else {
                    // 시간 선택 모드에서 확인 버튼 누르면 선택 완료
                    widget.onDateTimeChanged(_selectedDateTime);
                    Navigator.of(context).pop();
                  }
                },
                child: Text(
                  _currentMode == CupertinoDatePickerMode.date
                      ? widget.nextButtonText
                      : widget.confirmButtonText,
                  style: AppTextStyles.bodyMedium.copyWith(
                    color: AppColors.textSecondary,
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

// DateTimeWheelPicker를 BottomSheet로 표시하는 편의 함수
void showDateTimePickerBottomSheet({
  required BuildContext context,
  required WidgetRef ref,
  DateTime? initialDateTime,
  DateTime? minimumDate,
  DateTime? maximumDate,
  required Function(DateTime) onDateTimeChanged,
  String confirmButtonText = '확인',
  String cancelButtonText = '취소',
  String nextButtonText = '다음',
  CupertinoDatePickerMode initialMode = CupertinoDatePickerMode.date,
  String? title,
}) {
  // 초기 날짜가 null인 경우 현재 시간으로 설정
  DateTime effectiveInitialDate = initialDateTime ?? DateTime.now();

  // 허용 가능한 날짜 범위 설정 (2000-2025년)
  final minAllowedDate = DateTime(2000, 1, 1);
  final maxAllowedDate = DateTime(2025, 12, 31, 23, 59, 59);

  // 초기 날짜가 허용 범위를 벗어나는 경우 조정
  if (effectiveInitialDate.isBefore(minAllowedDate)) {
    effectiveInitialDate = minAllowedDate;
  } else if (effectiveInitialDate.isAfter(maxAllowedDate)) {
    effectiveInitialDate = DateTime(
      2025,
      12,
      31,
      effectiveInitialDate.hour,
      effectiveInitialDate.minute,
    );
  }

  // 년도가 범위를 벗어나는 경우 조정
  final year = effectiveInitialDate.year;
  if (year < 2000 || year > 2025) {
    effectiveInitialDate = DateTime(
      year < 2000 ? 2000 : 2025,
      effectiveInitialDate.month,
      effectiveInitialDate.day,
      effectiveInitialDate.hour,
      effectiveInitialDate.minute,
    );
  }

  showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (BuildContext context) {
      return Container(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(context).viewInsets.bottom,
        ),
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(context).size.height * 0.8,
        ),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(15)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(height: 20),
            // DateTimeWheelPicker
            DateTimeWheelPicker(
              initialDateTime: effectiveInitialDate,
              minimumDate: minimumDate,
              maximumDate: maximumDate,
              onDateTimeChanged: onDateTimeChanged,
              confirmButtonText: confirmButtonText,
              cancelButtonText: cancelButtonText,
              nextButtonText: nextButtonText,
              initialMode: initialMode,
            ),

            // 하단 여백
            SizedBox(height: 16),
          ],
        ),
      );
    },
  );
}
