import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:syncfusion_flutter_datepicker/datepicker.dart';

// DatePicker 상태를 관리하는 클래스
class DatePickerState {
  final bool isVisible; // 표시 여부
  final Offset? position; // 표시 위치
  final DateRangePickerSelectionMode selectionMode; // 선택 모드
  final PickerDateRange? selectedRange; // 선택된 범위
  final DateTime? selectedDate; // 선택된 날짜
  final List<DateTime>? selectedDates; // 선택된 날짜들
  final bool isConfirmed; // 확인 여부
  final String? lastUpdateKey; // 마지막 업데이트 키

  DatePickerState({
    this.isVisible = false,
    this.position,
    this.selectionMode = DateRangePickerSelectionMode.single,
    this.selectedRange,
    this.selectedDate,
    this.selectedDates,
    this.isConfirmed = false,
    this.lastUpdateKey,
  });

  // 상태 복사본 생성 메서드 (불변성 유지)
  DatePickerState copyWith({
    bool? isVisible,
    Offset? position,
    DateRangePickerSelectionMode? selectionMode,
    PickerDateRange? selectedRange,
    DateTime? selectedDate,
    List<DateTime>? selectedDates,
    bool? isConfirmed,
    String? lastUpdateKey,
  }) {
    return DatePickerState(
      isVisible: isVisible ?? this.isVisible,
      position: position ?? this.position,
      selectionMode: selectionMode ?? this.selectionMode,
      selectedRange: selectedRange ?? this.selectedRange,
      selectedDate: selectedDate ?? this.selectedDate,
      selectedDates: selectedDates ?? this.selectedDates,
      isConfirmed: isConfirmed ?? this.isConfirmed,
      lastUpdateKey: lastUpdateKey ?? this.lastUpdateKey,
    );
  }
}

// DatePicker 상태 Notifier
class DatePickerNotifier extends StateNotifier<DatePickerState> {
  DatePickerNotifier() : super(DatePickerState());

  // DatePicker 표시하기
  void showDatePicker({
    required Offset position,
    DateRangePickerSelectionMode? selectionMode,
    PickerDateRange? initialRange,
    DateTime? initialDate,
    List<DateTime>? initialDates,
  }) {
    state = state.copyWith(
      isVisible: true,
      position: position,
      selectionMode: selectionMode ?? DateRangePickerSelectionMode.single,
      selectedRange: initialRange,
      selectedDate: initialDate,
      selectedDates: initialDates,
      isConfirmed: false, // 초기화 시 확인 상태 초기화
    );
  }

  // DatePicker 숨기기
  void hideDatePicker() {
    // 이미 숨겨진 상태라면 상태 변경하지 않음
    if (!state.isVisible) return;

    state = state.copyWith(
        isVisible: false,
        isConfirmed: state.selectedRange != null || state.selectedDate != null);
  }

  // 선택된 날짜 범위 업데이트
  void updateSelectedRange(PickerDateRange range) {
    state = state.copyWith(
      selectedRange: range,
      isVisible: false,
      isConfirmed: true, // 확인 상태 true로 설정
    );
  }

  // 선택된 날짜 범위 업데이트 (중복 방지를 위한 키 사용)
  void updateSelectedRangeWithKey(PickerDateRange range, String updateKey) {
    // 마지막 업데이트 키와 동일하면 중복 업데이트 방지
    if (state.lastUpdateKey == updateKey) return;

    state = state.copyWith(
      selectedRange: range,
      isVisible: false,
      isConfirmed: true,
      lastUpdateKey: updateKey,
    );
  }

  // 선택된 날짜 업데이트
  void updateSelectedDate(DateTime date) {
    state = state.copyWith(
      selectedDate: date,
      isVisible: false,
      isConfirmed: true, // 확인 상태 true로 설정
    );
  }

  // 선택된 날짜들 업데이트
  void updateSelectedDates(List<DateTime> dates) {
    state = state.copyWith(
      selectedDates: dates,
      isVisible: false,
      isConfirmed: true, // 확인 상태 true로 설정
    );
  }

  // 확인 상태 초기화 메서드 추가
  void resetConfirmation() {
    state = state.copyWith(isConfirmed: false);
  }

  // 모든 선택 초기화 (필요한 경우 사용)
  void resetSelections() {
    state = state.copyWith(
      selectedRange: null,
      selectedDate: null,
      selectedDates: null,
      isConfirmed: false,
    );
  }

  // 마지막 선택 내용 초기화
  void clearLastSelection() {
    state = state.copyWith(
        selectedRange: null,
        selectedDate: null,
        selectedDates: null,
        isConfirmed: false);
  }

  void cancelSelection() {
    state = state.copyWith(isVisible: false, isConfirmed: false);
  }
}

// 전역 프로바이더 정의
final datePickerProvider =
    StateNotifierProvider<DatePickerNotifier, DatePickerState>((ref) {
  return DatePickerNotifier();
});
