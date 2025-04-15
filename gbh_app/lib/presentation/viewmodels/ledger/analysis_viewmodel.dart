import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart'
    show Reader, StateNotifier, StateNotifierProvider, Ref;
import 'package:intl/intl.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/data/repositories/ledger/ledger_repository.dart';
import 'package:marshmellow/data/models/ledger/category/withdrawal_category.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/chart/doughnut_chart_with_legend.dart'
    as doughnut;
import 'package:marshmellow/presentation/pages/ledger/widgets/chart/column_chart.dart'
    as column;
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/di/providers/calendar_providers.dart';

// 분석 페이지 상태 정의
class AnalysisState {
  final bool isLoading;
  final List<doughnut.ChartData> categoryChartData;
  final List<column.ChartData> weeklyChartData;
  final double totalExpenses;
  final String? errorMessage;
  final DateTime startDate;
  final DateTime endDate;

  AnalysisState({
    this.isLoading = false,
    this.categoryChartData = const [],
    this.weeklyChartData = const [],
    this.totalExpenses = 0,
    this.errorMessage,
    required this.startDate,
    required this.endDate,
  });

  AnalysisState copyWith({
    bool? isLoading,
    List<doughnut.ChartData>? categoryChartData,
    List<column.ChartData>? weeklyChartData,
    double? totalExpenses,
    String? errorMessage,
    DateTime? startDate,
    DateTime? endDate,
  }) {
    return AnalysisState(
      isLoading: isLoading ?? this.isLoading,
      categoryChartData: categoryChartData ?? this.categoryChartData,
      weeklyChartData: weeklyChartData ?? this.weeklyChartData,
      totalExpenses: totalExpenses ?? this.totalExpenses,
      errorMessage: errorMessage,
      startDate: startDate ?? this.startDate,
      endDate: endDate ?? this.endDate,
    );
  }
}

// 분석 뷰모델
class AnalysisViewModel extends StateNotifier<AnalysisState> {
  final LedgerRepository _repository;
  final Ref _ref;

  // 카테고리별 색상 리스트 - 상위 5개 항목에 사용
  final List<Color> _topCategoryColors = [
    AppColors.yellowPrimary,
    AppColors.pinkPrimary,
    AppColors.bluePrimary,
    AppColors.greenPrimary,
    AppColors.whiteLight,
  ];

  // 기타 항목의 색상
  final Color _otherCategoryColor = AppColors.blackPrimary;

  AnalysisViewModel(this._repository, this._ref)
      : super(AnalysisState(
          startDate: DateTime.now().subtract(const Duration(days: 30)),
          endDate: DateTime.now(),
        ));

  // 날짜 형식 변환 헬퍼 메서드
  String _formatDate(DateTime date) {
    return DateFormat('yyyyMMdd').format(date);
  }

  // 지출 분석 데이터 로드
  Future<void> loadAnalysisData({
    required DateTime startDate,
    required DateTime endDate,
  }) async {
    state = state.copyWith(
      isLoading: true,
      errorMessage: null,
      startDate: startDate,
      endDate: endDate,
    );

    try {
      // API 호출 파라미터 준비
      final formattedStartDate = _formatDate(startDate);
      final formattedEndDate = _formatDate(endDate);

      // 지출 내역만 조회 (WITHDRAWAL)
      final result = await _repository.getHouseholdByClassification(
        startDate: formattedStartDate,
        endDate: formattedEndDate,
        classification: 'WITHDRAWAL', // 지출 내역만 가져오기
      );

      // 트랜잭션 목록
      final transactions = result['transactions'] as List<Transaction>;

      // 카테고리별 데이터 처리
      final categoryData = _processCategoryData(transactions);

      // 주차별 데이터 처리
      final weeklyData = _processWeeklyData(transactions);

      // 총 지출 금액
      final totalExpenses = transactions.fold<double>(
          0, (sum, item) => sum + item.householdAmount);

      state = state.copyWith(
        isLoading: false,
        categoryChartData: categoryData,
        weeklyChartData: weeklyData,
        totalExpenses: totalExpenses,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
    }
  }

  // 카테고리별 데이터 처리
  List<doughnut.ChartData> _processCategoryData(
      List<Transaction> transactions) {
    // 카테고리별 합계 및 트랜잭션 목록 계산
    final Map<String, Map<String, dynamic>> categoryData = {};

    for (var transaction in transactions) {
      final category = transaction.householdCategory;

      if (!categoryData.containsKey(category)) {
        categoryData[category] = {
          'total': 0.0,
          'transactions': <Transaction>[],
        };
      }

      categoryData[category]!['total'] += transaction.householdAmount;
      categoryData[category]!['transactions'].add(transaction);
    }

    // 총액 계산
    final totalAmount =
        categoryData.values.fold<double>(0, (sum, data) => sum + data['total']);

    if (totalAmount <= 0) return [];

    // 퍼센트로 변환 및 내림차순 정렬
    List<MapEntry<String, Map<String, dynamic>>> categoryEntries = categoryData
        .entries
        .map((entry) => MapEntry(entry.key, {
              ...entry.value,
              'percent': (entry.value['total'] / totalAmount) * 100
            }))
        .toList()
      ..sort((a, b) => b.value['total'].compareTo(a.value['total']));

    // 차트 데이터 생성
    List<doughnut.ChartData> chartData = [];

    // 상위 5개 항목은 지정된 색상으로 차트 데이터 추가
    for (int i = 0; i < categoryEntries.length; i++) {
      if (i < 5) {
        final categoryEntry = categoryEntries[i];
        chartData.add(doughnut.ChartData(
          title: categoryEntry.key,
          value: categoryEntry.value['percent'],
          color: _topCategoryColors[i],
          amount: categoryEntry.value['total'],
          // 해당 카테고리의 모든 트랜잭션 함께 전달
          transactions:
              categoryEntry.value['transactions'] as List<Transaction>,
        ));
      } else {
        // 상위 5개 이외의 항목은 '기타'로 묶음
        double otherAmount = categoryEntries
            .skip(5)
            .fold(0.0, (sum, entry) => sum + entry.value['total']);

        double otherPercent = (otherAmount / totalAmount) * 100;

        // 기타 카테고리의 모든 트랜잭션도 포함
        List<Transaction> otherTransactions = categoryEntries
            .skip(5)
            .expand(
                (entry) => (entry.value['transactions'] as List<Transaction>))
            .toList();

        if (otherPercent > 0) {
          chartData.add(doughnut.ChartData(
            title: '기타',
            value: otherPercent,
            color: _otherCategoryColor,
            amount: otherAmount,
            transactions: otherTransactions,
          ));
        }
        break;
      }
    }

    return chartData;
  }

  // 주차별 데이터 처리 (월급일 기준 7일씩)
  List<column.ChartData> _processWeeklyData(List<Transaction> transactions) {
    // 결과를 담을 리스트
    List<column.ChartData> weeklyData = [];

    // 월급일 가져오기 (paydayProvider 사용)
    int payday = _ref.read(paydayProvider);

    // 선택된 기간 가져오기
    final selectedStartDate = state.startDate;
    final selectedEndDate = state.endDate;

    // 선택된 기간에 해당하는 월급 주기 계산
    DateTime periodStart;
    DateTime periodEnd;

    // 월급일 기준으로 시작일 계산
    if (selectedStartDate.day < payday) {
      // 선택된 시작일이 월급일보다 이전이면 전 달의 월급일부터
      periodStart =
          DateTime(selectedStartDate.year, selectedStartDate.month - 1, payday);
    } else {
      // 아니면 현재 달의 월급일부터
      periodStart =
          DateTime(selectedStartDate.year, selectedStartDate.month, payday);
    }

    // 시작일이 선택된 기간 시작일보다 이전이면 선택된 기간 시작일 사용
    if (periodStart.isBefore(selectedStartDate)) {
      periodStart = selectedStartDate;
    }

    // 다음 월급일 계산 (월급 주기의 끝)
    if (periodStart.month == 12) {
      periodEnd = DateTime(periodStart.year + 1, 1, payday)
          .subtract(const Duration(days: 1));
    } else {
      periodEnd = DateTime(periodStart.year, periodStart.month + 1, payday)
          .subtract(const Duration(days: 1));
    }

    // 끝일이 선택된 기간 끝일을 넘어가면 선택된 기간 끝일 사용
    if (periodEnd.isAfter(selectedEndDate)) {
      periodEnd = selectedEndDate;
    }

    // 월급일 기준으로 7일씩 구간 나누기
    final int totalDays = periodEnd.difference(periodStart).inDays + 1;
    final int weekCount = (totalDays / 7).ceil(); // 7일 단위로 올림 계산

    for (int weekIndex = 0; weekIndex < weekCount; weekIndex++) {
      // 각 주차의 시작일과 종료일 계산
      final weekStartDate = periodStart.add(Duration(days: weekIndex * 7));

      // 주차 종료일은 시작일+6 또는 종료일 중 더 빠른 날짜
      final calculatedEndDate = weekStartDate.add(const Duration(days: 6));
      final weekEndDate =
          calculatedEndDate.isAfter(periodEnd) ? periodEnd : calculatedEndDate;

      // 해당 주차의 거래 필터링
      final weekTransactions = transactions.where((transaction) {
        final txDate = transaction.dateTime;
        return txDate
                .isAfter(weekStartDate.subtract(const Duration(days: 1))) &&
            txDate.isBefore(weekEndDate.add(const Duration(days: 1)));
      }).toList();

      // 해당 주차의 지출 합계 계산
      double weekTotal = weekTransactions.fold<double>(
          0, (sum, transaction) => sum + transaction.householdAmount);

      // 주차 라벨 생성
      String weekLabel = '${weekIndex + 1}주차';

      // 차트 데이터 추가
      weeklyData.add(column.ChartData(
        label: weekLabel,
        value: weekTotal,
        color: AppColors.pinkPrimary,
      ));
    }

    return weeklyData;
  }

  // 이전 기간으로 이동
  void moveToPreviousPeriod() {
    final currentStartDate = state.startDate;
    final currentEndDate = state.endDate;

    // 기간의 길이 계산 (일수)
    final periodLength = currentEndDate.difference(currentStartDate).inDays;

    // 이전 기간 계산
    final newEndDate = currentStartDate.subtract(const Duration(days: 1));
    final newStartDate = newEndDate.subtract(Duration(days: periodLength));

    // 새 기간으로 데이터 로드
    loadAnalysisData(startDate: newStartDate, endDate: newEndDate);
  }

  // 다음 기간으로 이동
  void moveToNextPeriod() {
    final currentStartDate = state.startDate;
    final currentEndDate = state.endDate;

    // 기간의 길이 계산 (일수)
    final periodLength = currentEndDate.difference(currentStartDate).inDays;

    // 다음 기간 계산
    final newStartDate = currentEndDate.add(const Duration(days: 1));
    final newEndDate = newStartDate.add(Duration(days: periodLength));

    // 현재 날짜를 넘어가지 않도록 조정
    final today = DateTime.now();
    final adjustedEndDate = newEndDate.isAfter(today) ? today : newEndDate;

    // 새 기간으로 데이터 로드
    loadAnalysisData(startDate: newStartDate, endDate: adjustedEndDate);
  }
}

// 뷰모델 프로바이더 등록
final analysisViewModelProvider =
    StateNotifierProvider<AnalysisViewModel, AnalysisState>((ref) {
  final repository = ref.watch(ledgerRepositoryProvider);
  return AnalysisViewModel(repository, ref);
});
