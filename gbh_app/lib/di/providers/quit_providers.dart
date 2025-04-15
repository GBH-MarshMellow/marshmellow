import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/quit_api.dart';
import 'package:marshmellow/data/repositories/cookie/quit_repositary.dart';
import 'package:marshmellow/di/providers/api_providers.dart';

// API 프로바이더
final quitApiProvider = Provider<QuitApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return QuitApi(apiClient);
});

// Repository 프로바이더
final quitRepositoryProvider = Provider<QuitRepository>((ref) {
  final quitApi = ref.watch(quitApiProvider);
  return QuitRepository(quitApi);
});