import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/budget/wishlist_api.dart';
import 'package:marshmellow/data/models/wishlist/wishlist_model.dart';
import 'package:marshmellow/data/repositories/budget/wishlist_repository.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'dart:io';
import 'package:http_parser/http_parser.dart';
import 'package:path/path.dart' as path;

class WishlistState {
  final bool isLoading;
  final List<Wishlist> wishlists;
  final WishlistDetailResponse? selectedWishlist;
  final String? errorMessage;

  WishlistState({
    required this.isLoading,
    required this.wishlists,
    this.selectedWishlist,
    this.errorMessage,
  });

  factory WishlistState.initial() {
    return WishlistState(
      isLoading:false, 
      wishlists: [],
    );
  }

  WishlistState copywith({
    bool? isLoading,
    List<Wishlist>? wishlists,
    WishlistDetailResponse? selectedWishlist,
    String? errorMessage,
  }) {
    return WishlistState(
      isLoading: isLoading ?? this.isLoading,
      wishlists: wishlists ?? this.wishlists,
      selectedWishlist: selectedWishlist ?? this.selectedWishlist,
      errorMessage: errorMessage,
    );
  }
}

// 위시리스트 생성 상태 클래스
class WishlistCreationState {
  final bool isLoading;
  final WishlistCreationResponse? createdWishlist;
  final String? errorMessage;

  WishlistCreationState({
    required this.isLoading,
    this.createdWishlist,
    this.errorMessage,
  });

  factory WishlistCreationState.initial() {
    return WishlistCreationState(
      isLoading: false,
    );
  }

  WishlistCreationState copyWith({
    bool? isLoading,
    WishlistCreationResponse? createdWishlist,
    String? errorMessage,
  }) {
    return WishlistCreationState(
      isLoading: isLoading ?? this.isLoading,
      createdWishlist: createdWishlist ?? this.createdWishlist,
      errorMessage: errorMessage,
    );
  }
}

// 위시리스트 Notifier
class WishlistNotifier extends StateNotifier<WishlistState> {
  final WishlistRepository _repository;

  WishlistNotifier(this._repository) : super(WishlistState.initial());

  // 위시리스트 목록 조회
  Future<void> fetchWishlists() async {
    state = state.copywith(isLoading: true, errorMessage: null);

    try {
      final wishlists = await _repository.getWishlists();
      state = state.copywith(
        isLoading: false,
        wishlists: wishlists,
      );
    } catch (e) {
      state = state.copywith(
        isLoading: false,
        errorMessage: '위시리스트를 불러오는 중 오류가 발생했습니다: $e',
      );
    }
  }

  // 위시리스트 상세 조회
  Future<void> fetchWishlistDetail(int wishlistPk) async {
    state = state.copywith(isLoading: true, errorMessage: null);

    try {
      final wishlistDetail = await _repository.getWishlistDetail(wishlistPk);
      state = state.copywith(
        isLoading: false,
        selectedWishlist: wishlistDetail,
      );
    } catch (e) {
      state = state.copywith(
        isLoading: false,
        errorMessage: '위시리스트 상세 정보를 불러오는 중 오류가 발생했습니다: $e',
      );
    }
  }

  // 위시리스트 수정
  Future<void> updateWishlist({
    required int wishlistPk,
    String? productNickname,
    String? productName,
    int? productPrice,
    String? productImageUrl,
    String? productUrl,
  }) async {
    state = state.copywith(isLoading: true, errorMessage: null);

    try {
      await _repository.updateWishlist(
        wishlistPk: wishlistPk,
        productNickname: productNickname,
        productName: productName,
        productPrice: productPrice,
        productImageUrl: productImageUrl,
        productUrl: productUrl,
      );

      // 수정 후 목록 다시 불러오기
      await fetchWishlists();

      // 상세 정보도 업데이트
      if (state.selectedWishlist != null && state.selectedWishlist!.wishlistPk == wishlistPk) {
        await fetchWishlistDetail(wishlistPk);
      }
    } catch (e) {
      state = state.copywith(
        isLoading: false,
        errorMessage: '위시리스트를 수정하는 중 오류가 발생했습니다: $e',
      );
    }
  }

  // 위시리스트 삭제
  Future<void> deleteWishlist(int wishlistPk) async {
    state = state.copywith(isLoading: true, errorMessage: null);

    try {
      await _repository.deleteWishlist(wishlistPk);

      // 삭제된 항목 제외하고 목록 업데이트
      final updatedWishlists = state.wishlists
          .where((wishlist) => wishlist.wishlistPk != wishlistPk)
          .toList();

      state = state.copywith(
        isLoading: false,
        wishlists: updatedWishlists,
        selectedWishlist: state.selectedWishlist?.wishlistPk == wishlistPk 
            ? null 
            : state.selectedWishlist,
      );
    } catch (e) {
      state = state.copywith(
        isLoading: false,
        errorMessage: '위시리스트를 삭제하는 중 오류가 발생했습니다: $e',
      );
    }
  }
}

// 위시리스트 생성 Notifier
class WishlistCreationNotifier extends StateNotifier<WishlistCreationState> {
  final WishlistRepository _repository;
  final StateNotifierProviderRef _ref;

  WishlistCreationNotifier(this._repository, this._ref)
      : super(WishlistCreationState.initial());

  // 위시리스트 생성
  Future<void> createWishlist({
    required String productNickname,
    required String productName,
    required int productPrice,
    required String productUrl,
    File? imageFile,
  }) async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final apiClient = _ref.read(apiClientProvider);
      
      // FormData 생성
      FormData formData = FormData.fromMap({
        'productNickname': productNickname,
        'productName': productName,
        'productPrice': productPrice.toString(),
        'productUrl': productUrl,
      });
      
      // 이미지 파일이 있는 경우 추가
      if (imageFile != null) {
        final String fileName = path.basename(imageFile.path);
        final String fileExtension = path.extension(fileName).toLowerCase().substring(1);
        
        formData.files.add(MapEntry(
          'file',
          await MultipartFile.fromFile(
            imageFile.path,
            filename: fileName,
            contentType: MediaType('image', fileExtension),
          ),
        ));
      }
      
      // API 요청
      final response = await apiClient.post('/mm/wishlist', data: formData);
      
      // 응답 처리
      final wishlistResponse = WishlistResponse.fromJson(response.data);
      
      if (wishlistResponse.code == 200) {
        // 업데이트된 응답 형식에 맞춰 처리
        final responseData = wishlistResponse.data;
        final createdWishlist = WishlistCreationResponse(
          message: responseData['message'] ?? '',
          wishlistPk: responseData['wishlistPk'],
          productNickname: responseData['productNickname'],
          productName: responseData['productName'],
          productPrice: responseData['productPrice'],
          productImageUrl: responseData['productImageUrl'],
          productUrl: responseData['productUrl'],
        );
        
        state = state.copyWith(
          isLoading: false,
          createdWishlist: createdWishlist,
        );
        
        // 위시리스트 목록 갱신
        _ref.read(wishlistProvider.notifier).fetchWishlists();
      } else {
        throw Exception(wishlistResponse.message);
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: '위시리스트를 생성하는 중 오류가 발생했습니다: $e',
      );
    }
  }

  // URL 크롤링으로 상품 정보 가져오기
  Future<Map<String, dynamic>?> crawlProductUrl(String url) async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final data = await _repository.crawlProductUrl(url);
      state = state.copyWith(isLoading: false);
      return data;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: 'URL 정보를 가져오는 중 오류가 발생했습니다: $e',
      );
      return null;
    }
  }

  // 상태 초기화 (새로운 위시리스트 생성 시)
  void resetState() {
    state = WishlistCreationState.initial();
  }
}

// provider
final WishlistRepositoryProvider = Provider<WishlistRepository>((ref) {
  final wishlistApi = ref.watch(wishlistApiProvider);
  return WishlistRepository(wishlistApi);
});

final wishlistProvider = StateNotifierProvider<WishlistNotifier, WishlistState>((ref) {
  final repository = ref.watch(WishlistRepositoryProvider);
  return WishlistNotifier(repository);
});

final wishlistCreationProvider = StateNotifierProvider<WishlistCreationNotifier, WishlistCreationState>((ref) {
  final repository = ref.watch(WishlistRepositoryProvider);
  return WishlistCreationNotifier(repository, ref);
});

final wishlistApiProvider = Provider<WishlistApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return WishlistApi(apiClient);
});