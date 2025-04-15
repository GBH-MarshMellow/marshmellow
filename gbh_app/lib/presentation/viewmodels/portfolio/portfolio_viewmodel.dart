import 'dart:io';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_model.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_category_model.dart';
import 'package:marshmellow/data/repositories/cookie/portfolio_repositary.dart';
import 'package:marshmellow/di/providers/portfolio_providers.dart';

// 포트폴리오 상태 정의
class PortfolioState {
  final bool isLoading;
  final List<PortfolioModel> portfolios;
  final String? errorMessage;
  final Map<PortfolioCategoryModel, List<PortfolioModel>> groupedPortfolios;

  PortfolioState({
    this.isLoading = false,
    this.portfolios = const [],
    this.errorMessage,
    this.groupedPortfolios = const {},
  });

  PortfolioState copyWith({
    bool? isLoading,
    List<PortfolioModel>? portfolios,
    String? errorMessage,
    Map<PortfolioCategoryModel, List<PortfolioModel>>? groupedPortfolios,
  }) {
    return PortfolioState(
      isLoading: isLoading ?? this.isLoading,
      portfolios: portfolios ?? this.portfolios,
      errorMessage: errorMessage,
      groupedPortfolios: groupedPortfolios ?? this.groupedPortfolios,
    );
  }
}

// 포트폴리오 ViewModel
class PortfolioViewModel extends StateNotifier<PortfolioState> {
  final PortfolioRepository _repository;

  PortfolioViewModel(this._repository) : super(PortfolioState());

  // 포트폴리오 목록 로드
  Future<void> loadPortfolios() async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final portfolios = await _repository.getPortfolioList();
      final groupedPortfolios =
          _repository.groupPortfoliosByCategory(portfolios);

      state = state.copyWith(
        isLoading: false,
        portfolios: portfolios,
        groupedPortfolios: groupedPortfolios,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
    }
  }

  // 포트폴리오 등록
  Future<bool> createPortfolio({
    required dynamic file,
    required String portfolioMemo,
    required String fileName,
    required int portfolioCategoryPk,
  }) async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final newPortfolio = await _repository.createPortfolio(
        file: file,
        portfolioMemo: portfolioMemo,
        fileName: fileName,
        portfolioCategoryPk: portfolioCategoryPk,
      );

      // 기존 포트폴리오 목록에 새 항목 추가
      final updatedPortfolios = [...state.portfolios, newPortfolio];

      state = state.copyWith(
        isLoading: false,
        portfolios: updatedPortfolios,
      );

      return true;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
      return false;
    }
  }

  // 포트폴리오 수정
  Future<bool> updatePortfolio({
    required int portfolioPk,
    File? file,
    String? portfolioMemo,
    String? fileName,
    required int portfolioCategoryPk,
  }) async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final updatedPortfolio = await _repository.updatePortfolio(
        portfolioPk: portfolioPk,
        file: file,
        portfolioMemo: portfolioMemo,
        fileName: fileName,
        portfolioCategoryPk: portfolioCategoryPk,
      );

      // 기존 포트폴리오 목록에서 수정된 항목 찾아 업데이트
      final updatedPortfolios = state.portfolios.map((portfolio) {
        if (portfolio.portfolioPk == portfolioPk) {
          return updatedPortfolio;
        }
        return portfolio;
      }).toList();

      state = state.copyWith(
        isLoading: false,
        portfolios: updatedPortfolios,
      );

      return true;
    } catch (e, stackTrace) {
      print('Portfolio updatePortfolio error: $e');
      print('updatePortfolio stack trace: $stackTrace');

      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
      return false;
    }
  }

  // 특정 카테고리의 포트폴리오 목록 필터링
  List<PortfolioModel> getPortfoliosByCategory(int categoryPk) {
    return state.portfolios
        .where((portfolio) =>
            portfolio.portfolioCategory.portfolioCategoryPk == categoryPk)
        .toList();
  }

  // 포트폴리오 목록 삭제
  Future<bool> deletePortfolios({
    required List<int> portfolioPkList,
  }) async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final result = await _repository.deletePortfolios(
        portfolioPkList: portfolioPkList,
      );

      // 성공 시 해당 포트폴리오들을 목록에서 제거
      final updatedPortfolios = state.portfolios
          .where(
              (portfolio) => !portfolioPkList.contains(portfolio.portfolioPk))
          .toList();

      state = state.copyWith(
        isLoading: false,
        portfolios: updatedPortfolios,
      );

      return result;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
      return false;
    }
  }
}

// 포트폴리오 ViewModel 프로바이더
final portfolioViewModelProvider =
    StateNotifierProvider<PortfolioViewModel, PortfolioState>((ref) {
  final repository = ref.watch(portfolioRepositoryProvider);
  return PortfolioViewModel(repository);
});

// 포트폴리오 비동기 데이터 프로바이더
final portfoliosProvider = FutureProvider<List<PortfolioModel>>((ref) async {
  final repository = ref.watch(portfolioRepositoryProvider);
  return repository.getPortfolioList();
});
