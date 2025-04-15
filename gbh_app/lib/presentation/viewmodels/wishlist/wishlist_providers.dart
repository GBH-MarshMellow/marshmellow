import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/budget/wishlist_api.dart';
import 'package:marshmellow/data/repositories/budget/wishlist_repository.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'package:marshmellow/presentation/viewmodels/wishlist/wishlist_viewmodel.dart';

// 위시리스트 api 제공자   
/// API 클라이언트를 이용하여 WishlistApi 인스턴스를 생성
final wishlistApiProvider = Provider<WishlistApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return WishlistApi(apiClient);
});

// 위시리스트 repository 제공자
// wishlistApi를 이용하여 wishlistrepository 인스턴스를 생성
final wishlistRepositoryProvider = Provider<WishlistRepository>((ref) {
  final wishlistApi = ref.watch(wishlistApiProvider);
  return WishlistRepository(wishlistApi);
});

/// 위시리스트 State Notifier 제공자
/// 위시리스트 상태 관리를 위한 Notifier를 제공
final wishlistProvider = StateNotifierProvider<WishlistNotifier, WishlistState>((ref) {
  final repository = ref.watch(wishlistRepositoryProvider);
  return WishlistNotifier(repository);
});

/// 위시리스트 생성 Notifier 제공자
/// 위시리스트 생성 상태 관리를 위한 Notifier를 제공
final wishlistCreationProvider = 
    StateNotifierProvider<WishlistCreationNotifier, WishlistCreationState>((ref) {
  final repository = ref.watch(wishlistRepositoryProvider);
  return WishlistCreationNotifier(repository, ref);
});