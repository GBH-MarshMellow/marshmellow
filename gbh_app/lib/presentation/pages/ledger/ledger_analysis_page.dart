import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';

// 위젯
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/main/date_range_selector.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';

// 차트
import 'package:marshmellow/presentation/pages/ledger/widgets/chart/doughnut_chart_with_legend.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/chart/column_chart.dart'
    as column;

// 뷰모델
import 'package:marshmellow/presentation/viewmodels/ledger/analysis_viewmodel.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';

class LedgerAnalysisPage extends ConsumerStatefulWidget {
  const LedgerAnalysisPage({super.key});

  @override
  ConsumerState<LedgerAnalysisPage> createState() => _LedgerAnalysisPageState();
}

class _LedgerAnalysisPageState extends ConsumerState<LedgerAnalysisPage> {
  final _numberFormat = NumberFormat('#,###', 'ko_KR');
  bool _localLoading = false; // 로컬 로딩 상태

  @override
  void initState() {
    super.initState();
    // 페이지 로드 시 데이터 로드
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadAnalysisData();
    });
  }

  // 분석 데이터 로드
  Future<void> _loadAnalysisData() async {
    // 로컬 로딩 상태 활성화
    setState(() {
      _localLoading = true;
    });

    // DatePicker 상태 가져오기
    final datePickerState = ref.read(datePickerProvider);

    DateTime startDate;
    DateTime endDate;

    // DatePicker에서 선택된 날짜 범위가 있으면 사용
    if (datePickerState.selectedRange != null &&
        datePickerState.selectedRange!.startDate != null) {
      startDate = datePickerState.selectedRange!.startDate!;
      endDate = datePickerState.selectedRange!.endDate ?? startDate;
    } else {
      // 선택된 범위가 없으면 현재 월의 1일부터 오늘까지 사용
      final now = DateTime.now();
      startDate = DateTime(now.year, now.month, 1);
      endDate = now;
    }

    try {
      // 뷰모델에서 데이터 로드
      await ref.read(analysisViewModelProvider.notifier).loadAnalysisData(
            startDate: startDate,
            endDate: endDate,
          );
    } finally {
      // 로딩이 끝나면 로컬 로딩 상태 비활성화
      if (mounted) {
        setState(() {
          _localLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;
    final contentWidth = screenWidth * 0.9;
    final screenHeight = MediaQuery.of(context).size.height;

    // 분석 상태 구독
    final analysisState = ref.watch(analysisViewModelProvider);

    // 로딩 중 상태: 뷰모델의 isLoading 또는 로컬 로딩 상태
    final isLoading = analysisState.isLoading || _localLoading;

    // 날짜 범위 표시 문자열
    final dateFormatter = DateFormat('yy.MM.dd');
    final dateRangeText =
        '${dateFormatter.format(analysisState.startDate)} - ${dateFormatter.format(analysisState.endDate)}';

    return Scaffold(
      appBar: CustomAppbar(title: '지출 분석'),
      body: Stack(
        children: [
          // 메인 콘텐츠
          SafeArea(
            child: analysisState.errorMessage != null
                ? Center(
                    child: Text('오류가 발생했습니다: ${analysisState.errorMessage}'))
                : RefreshIndicator(
                    onRefresh: _loadAnalysisData,
                    child: SingleChildScrollView(
                      physics: const AlwaysScrollableScrollPhysics(),
                      child: Center(
                        child: Container(
                          width: contentWidth,
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              // 날짜 선택 컴포넌트
                              DateRangeSelector(
                                dateRange: dateRangeText,
                                enableDatePicker: false, // 날짜 선택기 비활성화
                                onPreviousPressed: () async {
                                  setState(() {
                                    _localLoading =
                                        true; // 이전 버튼 클릭 시 로딩 상태 활성화
                                  });

                                  try {
                                    ref
                                        .read(
                                            analysisViewModelProvider.notifier)
                                        .moveToPreviousPeriod();
                                    await _loadAnalysisData();
                                  } finally {
                                    if (mounted) {
                                      setState(() {
                                        _localLoading =
                                            false; // 로딩 완료 후 상태 비활성화
                                      });
                                    }
                                  }
                                },
                                onNextPressed: () async {
                                  setState(() {
                                    _localLoading =
                                        true; // 다음 버튼 클릭 시 로딩 상태 활성화
                                  });

                                  try {
                                    ref
                                        .read(
                                            analysisViewModelProvider.notifier)
                                        .moveToNextPeriod();
                                    await _loadAnalysisData();
                                  } finally {
                                    if (mounted) {
                                      setState(() {
                                        _localLoading =
                                            false; // 로딩 완료 후 상태 비활성화
                                      });
                                    }
                                  }
                                },
                                // onTap 속성 제거 (필요 없음)
                              ),

                              // 섹션 제목
                              Padding(
                                padding: const EdgeInsets.only(
                                    top: 10.0, bottom: 20.0),
                                child: Row(
                                  children: [
                                    Text(
                                      '총 지출:',
                                      style: AppTextStyles.bodyMedium,
                                    ),
                                    const SizedBox(width: 10),
                                    Text(
                                      '${_numberFormat.format(analysisState.totalExpenses.toInt())}원',
                                      style: AppTextStyles.bodyLarge,
                                    ),
                                  ],
                                ),
                              ),

                              // 도넛 차트와 범례 (통합 컴포넌트 사용)
                              DoughnutChartWithLegend(
                                data: analysisState.categoryChartData,
                                height: screenHeight * 0.35,
                                isLoading: isLoading,
                                ref: ref,
                              ),

                              const SizedBox(height: 24),

                              // 주차별 지출 섹션 (데이터가 있을 경우에만)
                              if (analysisState.weeklyChartData.isNotEmpty) ...[
                                Padding(
                                  padding: const EdgeInsets.only(
                                      top: 8.0, bottom: 16.0),
                                  child: Text(
                                    '주차별 지출',
                                    style: AppTextStyles.bodyMedium,
                                  ),
                                ),
                                SizedBox(
                                  height: screenHeight * 0.3,
                                  child: column.ColumnChart(
                                      data: analysisState.weeklyChartData),
                                ),
                              ],

                              // 여백을 위한 공간
                              const SizedBox(height: 24),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
          ),
        ],
      ),
    );
  }
}
