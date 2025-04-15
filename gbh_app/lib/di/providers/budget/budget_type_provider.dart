import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/budget/budget_avg_api.dart';
import 'package:marshmellow/data/datasources/remote/budget/budget_type_api.dart';
import 'package:marshmellow/data/repositories/budget/budget_type_repository.dart';
import 'package:marshmellow/di/providers/api_providers.dart';

/*
  예산 유형별 분석 API 프로바이더
  : 예산 유형별 분석 API를 호출하는 BudgetTypeApi를 제공하는 프로바이더입니다.
  : BudgetTypeApi는 ApiClient를 사용하여 실제 API 요청을 처리합니다.
*/
// 예산 유형별 분석 API 프로바이더
final budgetTypeApiProvider = Provider((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return BudgetTypeApi(apiClient);
});


final budgetAvgApiProvider = Provider((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return BudgetAvgApi(apiClient);
});

// 레포지토리 프로바이더
final budgetTypeRepositoryProvider = Provider<BudgetTypeRepository>((ref) {
  final typeApi = ref.watch(budgetTypeApiProvider);
  final avgApi = ref.watch(budgetAvgApiProvider);
  return BudgetTypeRepository(typeApi, avgApi);
});