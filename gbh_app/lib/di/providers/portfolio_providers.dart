import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/portfolio_api.dart';
import 'package:marshmellow/data/repositories/cookie/portfolio_repositary.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'package:marshmellow/presentation/viewmodels/portfolio/portfolio_viewmodel.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_model.dart';

// API 프로바이더
final portfolioApiProvider = Provider<PortfolioApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return PortfolioApi(apiClient);
});

// 리포지토리 프로바이더
final portfolioRepositoryProvider = Provider<PortfolioRepository>((ref) {
  final portfolioApi = ref.watch(portfolioApiProvider);
  return PortfolioRepository(portfolioApi);
});