/*
  위시 선택 상태
*/
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/repositories/budget/wish_repository.dart';
import 'package:marshmellow/data/repositories/budget/wishlist_repository.dart';
import 'package:marshmellow/presentation/viewmodels/wishlist/wish_provider.dart';
import 'package:marshmellow/presentation/viewmodels/wishlist/wishlist_providers.dart';

class WishSelectionState {
  final bool isLoading;
  final String? errorMessage;
  final bool isSuccess;

  WishSelectionState({
    required this.isLoading,
    this.errorMessage,
    required this.isSuccess
  });

  factory WishSelectionState.initial() {
    return WishSelectionState(
      isLoading: false, 
      isSuccess: false,
    );
  }

  WishSelectionState copyWith({
    bool? isLoading,
    String? errorMessage,
    bool? isSuccess,
  }) {
    return WishSelectionState(
      isLoading: isLoading ?? this.isLoading,
      errorMessage: errorMessage,
      isSuccess: isSuccess ?? this.isSuccess,
    );
  }
}

/*
  위시 선택 서비스 NOTIFIER
*/
class WishSelectionNotifier extends StateNotifier<WishSelectionState> {
  final WishRepository _wishRepository;
  final WishlistRepository _wishlistRepository;
  final StateNotifierProviderRef _ref;

  WishSelectionNotifier(
    this._wishRepository,
    this._wishlistRepository,
    this._ref,
  ) : super(WishSelectionState.initial());

  // 위시 선택 (진행중인 위시로 설정)
  Future<void> selectWish(int wishlistPk) async {
    state = state.copyWith(isLoading: true, errorMessage: null, isSuccess: false);
  
    try  {
      // 먼저 현재 진행중인 위시가 있다면 선택 해제
      final currentWish = await _wishRepository.getCurrentWish();
      if (currentWish != null) {
        await _unselectCurrentWish(currentWish.wishlistPk);
      }

      // 새 위시 선택하기
      await _selectNewWish(wishlistPk);

      // 성공 상태로 업데이트
      state = state.copyWith(
        isLoading: false,
        isSuccess: true,
      );

      // 위시 및 위시리스트 상태 갱신
      _ref.read(wishProvider.notifier).fetchCurrentWish();
      _ref.read(wishlistProvider.notifier).fetchWishlists();
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: '위시를 진행중으로 설정하는 중 오류가 발생했습니다. $e'
      );
    }
  }

  // 현재 진행중인 위시 선택 해제
  Future<void> unselectWish() async {
    state = state.copyWith(isLoading: true, errorMessage: null, isSuccess: false);

    try {
      // 현재 진행 중인 위시 확인
      final currentWish = await _wishRepository.getCurrentWish();
      
      if (currentWish != null) {
        await _unselectCurrentWish(currentWish.wishlistPk);
        
        // 성공 상태로 업데이트
        state = state.copyWith(
          isLoading: false,
          isSuccess: true,
        );

        // 위시 및 위시리스트 상태 갱신
        _ref.read(wishProvider.notifier).fetchCurrentWish();
        _ref.read(wishlistProvider.notifier).fetchWishlists();
      } else {
        state = state.copyWith(
          isLoading: false,
          errorMessage: '진행 중인 위시가 없습니다.',
        );
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: '위시 선택 해제 중 오류가 발생했습니다: $e',
      );
    }
  }

  // 내부적으로 현재 진행중인 위시 선택 해제
  Future<void> _unselectCurrentWish(int wishlistpk) async {
    await _wishlistRepository.updateWishlist(wishlistPk: wishlistpk, isSelected: 'N');
  }

  // 내부적으로 현재 진행중인 위시 선택 
  Future<void> _selectNewWish(int wishlistpk) async {
    await _wishlistRepository.updateWishlist(wishlistPk: wishlistpk, isSelected: 'Y');
  }
}

// 위시 선택 서비스 프로바이더
final WishSelectionProvider = StateNotifierProvider<WishSelectionNotifier, WishSelectionState>((ref) {
  final wishRepository = ref.watch(wishRepositoryProvider);
  final wishlistRepository = ref.watch(wishlistRepositoryProvider);
  return WishSelectionNotifier(wishRepository, wishlistRepository, ref);
});