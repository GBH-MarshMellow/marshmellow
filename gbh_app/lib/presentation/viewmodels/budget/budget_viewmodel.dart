import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/data/models/budget/budget_model.dart';
import 'package:marshmellow/data/repositories/budget/budget_repository.dart';
import 'package:marshmellow/di/providers/budget/budget_provider.dart';
import 'package:marshmellow/core/utils/widgets/widget_service.dart';

// Budget ViewModel 상태 클래스
class BudgetState {
  final List<BudgetModel> budgets;
  final BudgetModel? selectedBudget;
  final DailyBudgetModel? dailyBudget;
  final bool isLoading;
  final String? errorMessage;
  final int currentBudgetIndex;

  BudgetState({
    this.budgets = const [],
    this.selectedBudget,
    this.dailyBudget,
    this.isLoading = false,
    this.errorMessage,
    this.currentBudgetIndex = 0,
  });

  // 날짜 표시 텍스트 계산
  String get dateRangeText {
    if (selectedBudget == null) return '';

    final startDate = DateTime.parse(selectedBudget!.startDate);
    final endDate = DateTime.parse(selectedBudget!.endDate);

    final formatter = DateFormat('yyyy.MM.dd');
    return '${formatter.format(startDate)} - ${formatter.format(endDate)}';
  }

  BudgetState copyWith({
    List<BudgetModel>? budgets,
    BudgetModel? selectedBudget,
    DailyBudgetModel? dailyBudget,
    bool? isLoading,
    String? errorMessage,
    int? currentBudgetIndex,
  }) {
    return BudgetState(
      budgets: budgets ?? this.budgets,
      selectedBudget: selectedBudget ?? this.selectedBudget,
      dailyBudget: dailyBudget ?? this.dailyBudget,
      isLoading: isLoading ?? this.isLoading,
      errorMessage: errorMessage,
      currentBudgetIndex: currentBudgetIndex ?? this.currentBudgetIndex,
    );
  }
}

// Budget ViewModel 프로바이더
final budgetProvider =
    StateNotifierProvider.autoDispose<BudgetViewModel, BudgetState>((ref) {
  ref.keepAlive();
  final repository = ref.watch(budgetRepositoryProvider);
  return BudgetViewModel(repository);
});

// Budget ViewModel
class BudgetViewModel extends StateNotifier<BudgetState> {
  final BudgetRepository _repository;

  BudgetViewModel(this._repository) : super(BudgetState()) {
    // 초기 데이터 로딩
    fetchBudgets();
    fetchDailyBudget();
  }

  // 전체 예산 로드
  Future<void> fetchBudgets() async {
    try {
      state = state.copyWith(isLoading: true, errorMessage: null);
      final budgets = await _repository.getAllBudgets();
      print('📋 전체 예산 목록 로드 완료: ${budgets.length}개');

      // 예산이 있을 경우 첫 번째 예산을 선택
      BudgetModel? selectedBudget;

      if (budgets.isNotEmpty) {
        selectedBudget = budgets[0];

        print('📋 첫 번째 예산:');
        print('  - 예산 ID: ${budgets[0].budgetPk}');
        print('  - 금액: ${budgets[0].budgetAmount}');

        state = state.copyWith(
          budgets: budgets,
          selectedBudget: selectedBudget,
          isLoading: false,
          currentBudgetIndex: 0,
        );

        await fetchDailyBudget();
      } else {
        // 예산이 없는 경우 오늘의 예산은 로드하지 않음
        state = state.copyWith(
          budgets: budgets,
          selectedBudget: null,
          isLoading: false,
          currentBudgetIndex: 0,
        );
      }
    } catch (e) {
      // 404 오류는 예산이 없는 정상적인 상황
      if (e.toString().contains("404")) {
        state = state.copyWith(
          budgets: [],
          selectedBudget: null,
          isLoading: false,
          currentBudgetIndex: 0,
        );
      } else {
        state = state.copyWith(
          isLoading: false,
          errorMessage: e.toString(),
        );
      }
    }
  }

  // 일일 예산 로드
  Future<void> fetchDailyBudget() async {
    try {
      // 전체 예산이 없으면 오늘의 예산 로드 생략
      // if (state.budgets.isEmpty) return;

      final dailyBudget = await _repository.getDailyBudget();
      state = state.copyWith(dailyBudget: dailyBudget);

      // 홈 스크린 위젯 업데이트 로직 추가
      _updateWidget();
    } catch (e) {
      // 오늘의 예산 로드 실패 시 에러 메시지 표시하지 않음
      print('오늘의 예산 로드 실패: $e');
    }
  }

  // 위젯 업데이트 메서드
  Future<void> _updateWidget() async {
    try {
      if (!state.isLoading && state.dailyBudget != null) {
        // 오늘의 예산 금액 가져오기
        final dailyBudgetAmount = state.dailyBudget!.dailyBudgetAmount;
        // 위젯 서비스를 통해 홈 스크린 위젯 업데이트
        await WidgetService.updateBudgetWidget(dailyBudgetAmount);
        print('📱 위젯 업데이트 요청 (BudgetViewModel): $dailyBudgetAmount원');
      } else {
        print('⚠️ 위젯 업데이트 실패: 일일 예산 정보 없음');
      }
    } catch (e) {
      print('⚠️ 위젯 업데이트 오류: $e');
    }
  }

  // 이전 예산으로 이동
  void navigateToPreviousBudget() {
    if (state.budgets.isEmpty || state.currentBudgetIndex <= 0) return;

    final newIndex = state.currentBudgetIndex - 1;
    final newSelectedBudget = state.budgets[newIndex];

    state = state.copyWith(
      selectedBudget: newSelectedBudget,
      currentBudgetIndex: newIndex,
    );

    // 예산이 변경되면 일일 예산도 다시 로드
    fetchDailyBudget();
  }

  // 다음 예산으로 이동
  void navigateToNextBudget() {
    if (state.budgets.isEmpty ||
        state.currentBudgetIndex >= state.budgets.length - 1) return;

    final newIndex = state.currentBudgetIndex + 1;
    final newSelectedBudget = state.budgets[newIndex];

    state = state.copyWith(
      selectedBudget: newSelectedBudget,
      currentBudgetIndex: newIndex,
    );

    // 예산이 변경되면 일일 예산도 다시 로드
    fetchDailyBudget();
  }

  // 예산 카테고리 업데이트
  Future<void> updateBudgetCategory(int budgetCategoryPk, int newAmount) async {
    try {
      state = state.copyWith(isLoading: true, errorMessage: null);
      await _repository.updateBudgetCategory(budgetCategoryPk, newAmount);

      // 업데이트 후 모든 예산 다시 로드
      await fetchBudgets();
      // 일일 예산도 다시 로드
      await fetchDailyBudget();
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
    }
  }

  // 예산 알람 시간 업데이트
  Future<void> updateBudgetAlarm(String alarmTime) async {
    try {
      state = state.copyWith(isLoading: true, errorMessage: null);
      await _repository.updateBudgetAlarm(alarmTime);
      state = state.copyWith(isLoading: false);
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
    }
  }

  // 예산 생성
  Future<Map<String, dynamic>> createBudget({
    required int salary,
    required double fixedExpense,
    required double foodExpense,
    required double transportationExpense,
    required double marketExpense,
    required double financialExpense,
    required double leisureExpense,
    required double coffeeExpense,
    required double shoppingExpense,
    required double emergencyExpense,
  }) async {
    try {
      state = state.copyWith(isLoading: true, errorMessage: null);

      final result = await _repository.createBudget(
        salary: salary,
        fixedExpense: fixedExpense,
        foodExpense: foodExpense,
        transportationExpense: transportationExpense,
        marketExpense: marketExpense,
        financialExpense: financialExpense,
        leisureExpense: leisureExpense,
        coffeeExpense: coffeeExpense,
        shoppingExpense: shoppingExpense,
        emergencyExpense: emergencyExpense,
      );

      // 예산 생성 후 전체 예산 다시 로드
      await fetchBudgets();

      state = state.copyWith(isLoading: false);
      return result;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: '예산 생성에 실패했습니다: $e',
      );
      rethrow;
    }
  }
}
