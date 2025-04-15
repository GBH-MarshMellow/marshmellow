import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/models/cookie/quit/quit_model.dart';
import 'package:marshmellow/data/repositories/cookie/quit_repositary.dart';
import 'package:marshmellow/di/providers/quit_providers.dart';

class QuitState {
  final bool isLoading;
  final AverageSpendingData? averageSpending;
  final String? errorMessage;
  final DelusionData? availableAmount;

  QuitState({
    this.isLoading = false,
    this.averageSpending,
    this.errorMessage,
    this.availableAmount,
  });

  QuitState copyWith({
    bool? isLoading,
    AverageSpendingData? averageSpending,
    String? errorMessage,
    DelusionData? availableAmount,
  }) {
    return QuitState(
      isLoading: isLoading ?? this.isLoading,
      averageSpending: averageSpending ?? this.averageSpending,
      errorMessage: errorMessage,
      availableAmount: availableAmount ?? this.availableAmount,
    );
  }
}

class QuitViewModel extends StateNotifier<QuitState> {
  final QuitRepository _repository;

  QuitViewModel(this._repository) : super(QuitState());

  Future<void> loadAverageSpending() async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final data = await _repository.getAverageSpending();
      state = state.copyWith(isLoading: false, averageSpending: data);
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: '평균 지출 데이터를 불러오는데 실패했습니다: $e',
      );
    }
  }

  // 퇴사 망상 - 사용 가능 금액 조회
  Future<void> loadAvailableAmount() async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final data = await _repository.getAvailableAmount();
      state = state.copyWith(isLoading: false, availableAmount: data);
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: '사용 가능 금액 데이터를 불러오는데 실패했습니다: $e',
      );
    }
  }

  // 모든 데이터를 한 번에 로드하는 메서드
  Future<void> loadAllData() async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      // 병렬로 API 호출
      final averageSpendingFuture = _repository.getAverageSpending();
      final availableAmountFuture = _repository.getAvailableAmount();

      // 모든 Future가 완료될 때까지 대기
      final results = await Future.wait([
        averageSpendingFuture,
        availableAmountFuture,
      ]);

      // 결과 저장
      state = state.copyWith(
        isLoading: false,
        averageSpending: results[0] as AverageSpendingData,
        availableAmount: results[1] as DelusionData,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: '데이터를 불러오는데 실패했습니다: $e',
      );
    }
  }
}

// Provider 등록
final quitViewModelProvider =
    StateNotifierProvider<QuitViewModel, QuitState>((ref) {
  final repository = ref.watch(quitRepositoryProvider);
  return QuitViewModel(repository);
});
