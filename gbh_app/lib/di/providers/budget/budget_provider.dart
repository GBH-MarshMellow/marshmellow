import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/budget/budget_api.dart';
import 'package:marshmellow/data/repositories/budget/budget_repository.dart';
import 'package:marshmellow/di/providers/api_providers.dart';

/*
  예산 api 프로바이더
*/
final budgetApiProvider = Provider<BudgetApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return BudgetApi(apiClient);
});

/*
  예산 레포지토리 프로바이더
*/
final budgetRepositoryProvider = Provider<BudgetRepository>((ref) {
  final budgetApi = ref.watch(budgetApiProvider);
  return BudgetRepository(budgetApi);
});