import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/data/repositories/ledger/ledger_repository.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';

// 검색 상태 정의
class SearchState {
  final bool isLoading;
  final List<Transaction> searchResults;
  final String? errorMessage;
  final String searchTerm;

  SearchState({
    this.isLoading = false,
    this.searchResults = const [],
    this.errorMessage,
    this.searchTerm = '',
  });

  SearchState copyWith({
    bool? isLoading,
    List<Transaction>? searchResults,
    String? errorMessage,
    String? searchTerm,
  }) {
    return SearchState(
      isLoading: isLoading ?? this.isLoading,
      searchResults: searchResults ?? this.searchResults,
      errorMessage: errorMessage,
      searchTerm: searchTerm ?? this.searchTerm,
    );
  }
}

// 검색 ViewModel
class SearchViewModel extends StateNotifier<SearchState> {
  final LedgerRepository _repository;

  SearchViewModel(this._repository) : super(SearchState());

  // 검색 실행
  Future<void> search({
    required String keyword,
    required String startDate,
    required String endDate,
  }) async {
    if (keyword.trim().isEmpty) {
      state = state.copyWith(
        searchResults: [],
        searchTerm: '',
        errorMessage: null,
      );
      return;
    }

    state = state.copyWith(
      isLoading: true,
      searchTerm: keyword,
      errorMessage: null,
    );

    try {
      print('검색 시작: $keyword, 기간: $startDate-$endDate');

      final results = await _repository.searchTransactions(
        startDate: startDate,
        endDate: endDate,
        keyword: keyword,
      );

      print('검색 결과: ${results.length}개 항목');

      state = state.copyWith(
        isLoading: false,
        searchResults: results,
      );
    } catch (e) {
      print('검색 오류: $e');
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
    }
  }

  // 검색 결과 초기화
  void clearSearch() {
    state = state.copyWith(
      searchResults: [],
      searchTerm: '',
      errorMessage: null,
    );
  }
}

// Provider 등록 - 클래스 외부로 이동
final searchViewModelProvider =
    StateNotifierProvider<SearchViewModel, SearchState>((ref) {
  final repository = ref.watch(ledgerRepositoryProvider);
  return SearchViewModel(repository);
});
