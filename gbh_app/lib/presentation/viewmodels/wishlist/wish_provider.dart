import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/budget/wishlist_api.dart';
import 'package:marshmellow/data/repositories/budget/wish_repository.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'package:marshmellow/presentation/viewmodels/wishlist/wish_viewmodel.dart';

// Wish API 제공자
final wishApiProvider = Provider<WishApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return WishApi(apiClient);
});

// Wish Repository 제공자
final wishRepositoryProvider = Provider<WishRepository>((ref) {
  final wishApi = ref.watch(wishApiProvider);
  return WishRepository(wishApi);
});

// Wish State Notifier 제공자
final wishProvider = StateNotifierProvider<WishNotifier, WishState>((ref) {
  final repository = ref.watch(wishRepositoryProvider);
  return WishNotifier(repository);
});