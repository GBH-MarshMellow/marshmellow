import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/data/models/my/user_detail_info.dart';
import 'package:marshmellow/data/models/wishlist/wish_model.dart';
import 'package:marshmellow/data/repositories/budget/wish_repository.dart';
import 'package:marshmellow/di/providers/core_providers.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_info_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/wishlist/wish_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

class WishSelectionState {
  final bool isLoading;
  final bool isSuccess;
  final List<DemDepItem> accounts;
  final String? errorMessage;

  WishSelectionState({
    required this.isLoading,
    required this.isSuccess,
    required this.accounts,
    this.errorMessage,
  });

  factory WishSelectionState.initial() {
    return WishSelectionState(
      isLoading: false,
      isSuccess: false,
      accounts: [],
    );
  }

  WishSelectionState copyWith({
    bool? isLoading,
    bool? isSuccess,
    List<DemDepItem>? accounts,
    String? errorMessage,
  }) {
    return WishSelectionState(
      isLoading: isLoading ?? this.isLoading,
      isSuccess: isSuccess ?? this.isSuccess,
      accounts: accounts ?? this.accounts,
      errorMessage: errorMessage,
    );
  }
}

class WishSelectionNotifier extends StateNotifier<WishSelectionState> {
  final WishRepository _wishRepository;
  
  WishSelectionNotifier(this._wishRepository)
      : super(WishSelectionState.initial());

  // 입출금 계좌 목록 조회
  Future<void> fetchDemDepList() async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final response = await _wishRepository.getDemDepList();
      if (response.code == 200) {
        state = state.copyWith(
          isLoading: false,
          accounts: response.data.demandDepositList,
        );
      } else {
        throw Exception(response.message);
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: '입출금 계좌 목록을 불러오는 중 오류가 발생했습니다: $e',
      );
    }
  }

  // 위시 선택 및 자동이체 등록 통합 메서드
  Future<void> selectWishAndCreateAutoTransfer({
    required int wishlistPk,
    required String withdrawalAccountNo,
    required String depositAccountNo,
    required String dueDate,
    required int transactionBalance,
  }) async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      // 1. 위시 선택 API 호출
      final selectResponse = await _wishRepository.selectWish(wishlistPk, "Y");
      if (selectResponse.code != 200) {
        throw Exception(selectResponse.message);
      }

      // 2. 자동이체 등록 API 호출
      final prefs = await SharedPreferences.getInstance();
      final userPk =prefs.getString(StorageKeys.userId) ?? '2';
      print('🍡🍡🍡저는 $userPk 번 사용자입니다');
      // final userPk = 1; // TODO: 실제 사용자 PK 가져오기 (인증 관련 provider에서 가져오는 것이 좋음)
      
      final transferResponse = await _wishRepository.registerAutoTransfer(
        withdrawalAccountNo: withdrawalAccountNo,
        depositAccountNo: depositAccountNo,
        dueDate: dueDate,
        transactionBalance: transactionBalance,
        wishListPk: wishlistPk,
        userPk: int.parse(userPk),
      );
      
      if (transferResponse.code != 200) {
        throw Exception(transferResponse.message);
      }
      
      state = state.copyWith(
        isLoading: false,
        isSuccess: true,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: '위시 등록 중 오류가 발생했습니다: $e',
      );
      rethrow;
    }
  }

  // 상태 초기화
  void reset() {
    state = WishSelectionState.initial();
  }
}

// Provider
final wishSelectionProvider = StateNotifierProvider.autoDispose<WishSelectionNotifier, WishSelectionState>((ref) {
  final repository = ref.watch(wishRepositoryProvider);
  return WishSelectionNotifier(repository);
});