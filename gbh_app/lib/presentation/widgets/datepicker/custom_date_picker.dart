import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';

import 'package:intl/intl.dart'; // 날짜 포맷팅을 위한 패키지
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:syncfusion_flutter_datepicker/datepicker.dart'; // 캘린더 기능 제공 패키지
import 'package:intl/date_symbol_data_local.dart';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:marshmellow/di/providers/my/salary_provider.dart';
import 'package:marshmellow/di/providers/calendar_providers.dart';

/// 커스텀 날짜 선택 위젯
/// 여러 날짜 선택 모드를 지원하는 재사용 가능한 날짜 선택 컴포넌트
class CustomDatePicker extends ConsumerStatefulWidget {
  // 날짜 선택 변경 시 호출될 콜백 함수
  final Function(DateRangePickerSelectionChangedArgs)? onSelectionChanged;
  // 확인 버튼 콜백
  final Function(PickerDateRange)? onConfirm;
  // 취소 버튼 콜백
  final VoidCallback? onCancel;
  // 날짜 선택 모드 (단일 날짜, 다중 날짜, 범위 등)
  final DateRangePickerSelectionMode selectionMode;
  // 초기 선택 범위 (시작일-종료일)
  final PickerDateRange? initialSelectedRange;
  // 초기 선택 날짜 (단일 선택 모드용)
  final DateTime? initialSelectedDate;
  // 초기 선택 날짜들 (다중 선택 모드용)
  final List<DateTime>? initialSelectedDates;

  // 생성자: 위젯이 생성될 때 받는 파라미터들
  const CustomDatePicker({
    super.key, // 위젯 식별을 위한 키
    this.onSelectionChanged, // 선택 변경 시 호출될 함수
    this.onConfirm, // 확인 버튼 콜백
    this.onCancel, // 취소 버튼 콜백
    this.selectionMode = DateRangePickerSelectionMode.single, // 기본값은 단일 선택 모드
    this.initialSelectedRange, // 초기 범위 (null 가능)
    this.initialSelectedDate, // 초기 날짜 (null 가능)
    this.initialSelectedDates, // 초기 날짜들 (null 가능)
  });

  @override
  // StatefulWidget은 State 객체를 생성해야 함
  // 이 State 객체가 위젯의 상태를 관리함
  ConsumerState<CustomDatePicker> createState() => CustomDatePickerState();
}

/// CustomDatePicker의 상태 관리 클래스
/// 위젯의 내부 상태(선택된 날짜 등)를 관리하고 UI를 구성함
class CustomDatePickerState extends ConsumerState<CustomDatePicker> {
  // 선택된 날짜 정보를 저장할 변수들
  String _selectedDate = ''; // 단일 선택된 날짜
  String _dateCount = ''; // 다중 선택된 날짜 개수
  String _range = ''; // 선택된 날짜 범위 (시작-종료)
  String _rangeCount = ''; // 선택된 범위 개수

  // 현재 선택된 날짜 범위
  PickerDateRange? _currentRange;

  // BorderRadius 값을 변수로 추출
  final double _borderRadius = 5.0;

  @override
  void initState() {
    super.initState();
    // 한국어 로케일 초기화
    initializeDateFormatting('ko_KR', null);
    // 오늘 날짜로 _range 초기화
    final DateTime today = DateTime.now();
    _range =
        '${DateFormat('yy/MM/dd').format(today)} - ${DateFormat('yy/MM/dd').format(today)}';

    // 초기 선택 범위가 있으면 설정
    if (widget.initialSelectedRange != null) {
      _currentRange = widget.initialSelectedRange;
      _updateRangeText(_currentRange!);
    }
  }

  // 범위 텍스트 업데이트 함수
  void _updateRangeText(PickerDateRange range) {
    final startDate = range.startDate;
    final endDate = range.endDate ?? range.startDate;
    // null 체크 추가
    if (startDate == null) {
      _range = '날짜를 선택하세요';
      return;
    }

    if (endDate == null) {
      _range = '${DateFormat('yy/MM/dd').format(startDate)}';
      return;
    }
    if (startDate.year == endDate.year &&
        startDate.month == endDate.month &&
        startDate.day == endDate.day) {
      _range = '${DateFormat('yy/MM/dd').format(startDate)}';
    } else {
      _range = '${DateFormat('yy/MM/dd').format(startDate)} -'
          ' ${DateFormat('yy/MM/dd').format(endDate)}';
    }
  }

  /// 날짜 선택 변경 이벤트 핸들러
  /// 사용자가 날짜를 선택할 때마다 호출됨
  void _onSelectionChanged(DateRangePickerSelectionChangedArgs args) {
    // setState: 상태가 변경되었음을 Flutter에 알리고 UI를 다시 그리도록 함
    setState(() {
      // args.value: 선택된 날짜 정보를 담고 있음
      // 선택 모드에 따라 다른 타입의 데이터가 들어옴

      // 1. 범위 선택 모드인 경우
      if (args.value is PickerDateRange) {
        _currentRange = args.value;
        _updateRangeText(_currentRange!);
      }
      // 2. 단일 날짜 선택 모드인 경우
      else if (args.value is DateTime) {
        // 선택된 날짜를 포맷팅하여 저장
        _selectedDate = DateFormat('yy/MM/dd').format(args.value);
      }
      // 3. 다중 날짜 선택 모드인 경우
      else if (args.value is List<DateTime>) {
        // 선택된 날짜 개수를 저장
        _dateCount = args.value.length.toString();
      }
      // 4. 다중 범위 선택 모드인 경우
      else {
        // 선택된 범위 개수를 저장
        _rangeCount = args.value.length.toString();
      }
    });

    // 외부에서 전달받은 콜백 함수가 있으면 호출
    // 부모 위젯에서 날짜 선택 변경 이벤트를 처리할 수 있도록 함
    if (widget.onSelectionChanged != null) {
      widget.onSelectionChanged!(args);
    }
  }

  @override
  // 위젯의 UI를 구성하는 메서드
  Widget build(BuildContext context) {
    return Container(
      width: MediaQuery.of(context).size.width * 0.9, // 화면의 90%
      decoration: BoxDecoration(
        color: AppColors.whiteLight,
        borderRadius: BorderRadius.circular(_borderRadius),
        border: Border.all(
          color: AppColors.blackPrimary, // 테두리 색상
          width: 0.5, // 테두리 두께
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min, // 내용물에 맞게 크기 조정
        children: <Widget>[
          // 1. 선택된 정보 표시 부분
          Container(
            width: double.infinity, // 상위의 100%
            padding: const EdgeInsets.symmetric(
                horizontal: 10.0, vertical: 8.0), // 여백 설정
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start, // 왼쪽 정렬
              children: <Widget>[
                // 선택된 날짜, 개수, 범위 정보를 텍스트로 표시
                Text(
                  _range,
                  style: AppTextStyles.bodyMediumLight.copyWith(
                    fontWeight: FontWeight.w600,
                    decoration: TextDecoration.none,
                  ),
                ), // 선택된 범위
              ],
            ),
          ),
          Divider(
            height: 1,
            thickness: 0.7,
            color: AppColors.blackPrimary.withOpacity(0.3),
          ),
          // 2. 실제 날짜 선택기 위젯
          // Expanded를 Flexible로 변경하거나 크기 제한
          Container(
            height: 350, // 적절한 고정 높이 설정
            child: SfDateRangePicker(
              // 날짜 선택 변경 이벤트 핸들러 설정
              onSelectionChanged: _onSelectionChanged,

              // 선택 모드 설정 (단일, 다중, 범위 등)
              // widget.속성: 부모 위젯(StatefulWidget)에서 전달받은 속성에 접근
              selectionMode: widget.selectionMode,

              // 초기 선택 값들 설정
              initialSelectedRange: widget.initialSelectedRange,
              initialSelectedDate: widget.initialSelectedDate,
              initialSelectedDates: widget.initialSelectedDates,
              backgroundColor: AppColors.whiteLight,

              // 헤더 형식 설정
              monthFormat: 'M월',
              // 화살표 표시 설정
              showNavigationArrow: true,

              // 다른 헤더 스타일은 그대로 유지
              headerStyle: DateRangePickerHeaderStyle(
                backgroundColor: AppColors.whiteLight,
                textStyle: AppTextStyles.bodyMedium.copyWith(
                  fontWeight: FontWeight.w500,
                ),
              ),

              selectionColor: AppColors.textPrimary, // 단일 선택 시 색상
              startRangeSelectionColor: AppColors.textPrimary, // 범위 선택 시작일 색상
              endRangeSelectionColor: AppColors.textPrimary, // 범위 선택 종료일 색상
              rangeSelectionColor:
                  AppColors.blackPrimary.withOpacity(0.3), // 범위 내 날짜들의 색상
              // 오늘 날짜의 강조 색상
              todayHighlightColor: AppColors.blackPrimary,

              // 오늘 날짜의 텍스트 스타일 적용
              monthCellStyle: DateRangePickerMonthCellStyle(
                todayTextStyle: AppTextStyles.bodySmall.copyWith(
                  color: AppColors.blackPrimary,
                ),
                // 오늘 날짜의 셀 꾸미기
                todayCellDecoration: BoxDecoration(
                  shape: BoxShape.circle, // 원형 모양
                  color: Colors.transparent, // 배경색
                  border: Border.all(
                    color: AppColors.blackPrimary, // 테두리 색상
                    width: 1, // 테두리 두께
                  ),
                ),
              ),
            ),
          ),
          Container(
            width: double.infinity,
            height: 40,
            padding: EdgeInsets.fromLTRB(16.0, 0.0, 16.0, 0.0), // 패딩 수정
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween, // 정렬 수정
              children: [
                // 초기화 버튼 추가
                TextButton(
                  onPressed: () {
                    // 초기화 버튼 클릭 시
                    ref.read(datePickerProvider.notifier).clearLastSelection();

                    // 월급날 기준으로 설정
                    final payday = ref.read(paydayProvider);
                    final now = DateTime.now();

                    DateTime startDate;
                    DateTime endDate;

                    // 현재 날짜가 월급일 이전이면 전 달의 월급일부터
                    if (now.day < payday) {
                      startDate = DateTime(now.year, now.month - 1, payday);
                      endDate = DateTime(now.year, now.month, payday - 1);
                    } else {
                      // 현재 날짜가 월급일 이후면 현재 달의 월급일부터
                      startDate = DateTime(now.year, now.month, payday);

                      // 다음 달의 월급일 이전 날까지
                      if (startDate.month == 12) {
                        endDate = DateTime(startDate.year + 1, 1, payday - 1);
                      } else {
                        endDate = DateTime(now.year, now.month + 1, payday - 1);
                      }
                    }

                    // 새 범위 설정 및 picker 닫기
                    ref.read(datePickerProvider.notifier).updateSelectedRange(
                        PickerDateRange(startDate, endDate));
                  },
                  style: TextButton.styleFrom(
                    padding: EdgeInsets.zero,
                    minimumSize: Size.zero,
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                    overlayColor: AppColors.buttonBlack,
                  ),
                  child: Text(
                    '초기화',
                    style:
                        TextStyle(fontSize: 14.0, color: AppColors.buttonBlack),
                  ),
                ),

                // 오른쪽 버튼들
                Row(
                  children: [
                    TextButton(
                      onPressed: () {
                        // 취소 버튼 클릭 시
                        ref.read(datePickerProvider.notifier).hideDatePicker();
                        if (widget.onCancel != null) {
                          widget.onCancel!();
                        }
                      },
                      style: TextButton.styleFrom(
                        padding: EdgeInsets.zero,
                        minimumSize: Size.zero,
                        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                        overlayColor: AppColors.buttonBlack,
                      ),
                      child: Text(
                        '취소',
                        style: TextStyle(
                            fontSize: 14.0, color: AppColors.buttonBlack),
                      ),
                    ),
                    SizedBox(width: 30),
                    TextButton(
                      onPressed: () {
                        // 확인 버튼 클릭 시
                        ref.read(datePickerProvider.notifier).hideDatePicker();
                        if (widget.onConfirm != null && _currentRange != null) {
                          widget.onConfirm!(_currentRange!);
                        }
                      },
                      style: TextButton.styleFrom(
                        padding: EdgeInsets.zero,
                        minimumSize: Size.zero,
                        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                        overlayColor: AppColors.buttonBlack,
                      ),
                      child: Text(
                        '확인',
                        style: TextStyle(
                            fontSize: 14.0, color: AppColors.buttonBlack),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
