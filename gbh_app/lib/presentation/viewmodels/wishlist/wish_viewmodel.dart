import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/models/wishlist/wish_model.dart';
import 'package:marshmellow/data/repositories/budget/wish_repository.dart';

class WishState {
  final bool isLoading;
  final WishDetail? currentWish;
  final String? errorMessage;

  WishState({
    required this.isLoading,
    this.currentWish,
    this.errorMessage,
  });

  factory WishState.initial() {
    return WishState(isLoading: false);
  }

  WishState copyWith({
    bool? isLoading,
    WishDetail? currentWish,
    String? errorMessage,
  }) {
    return WishState(
      isLoading: isLoading ?? this.isLoading,
      currentWish: currentWish ?? this.currentWish,
      errorMessage: errorMessage,
    );
  }
}

class WishNotifier extends StateNotifier<WishState> {
  final WishRepository _repository;

  WishNotifier(this._repository) : super(WishState.initial());

  // 현재 진행 중인 wish 조회
  Future<void> fetchCurrentWish() async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final wish = await _repository.getCurrentWish();
      print('Wish received: $wish');
      // wish 가 null 이더라도 에러가 아닌 정상 응답으로 처리
      state = state.copyWith(
        isLoading: false,
        currentWish: wish,
        errorMessage: null,
      );
      print('State updated with wish: ${state.currentWish}');
    } catch (e) {
      print('Error fetching wish: $e');
      state = state.copyWith(
        isLoading: false,
        errorMessage: '현재 진행 중인 wish를 불러오는 중 오류가 발생했습니다: $e',
      );
    }
  }
}