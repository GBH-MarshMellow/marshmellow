import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/api_client.dart';
import 'package:marshmellow/data/models/budget/transaction_model_adaptor.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/di/providers/api_providers.dart';

// Provider for the category transaction API
final categoryTransactionApiProvider = Provider<CategoryTransactionApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return CategoryTransactionApi(apiClient);
});

// Provider for the repository
final categoryTransactionRepositoryProvider = Provider<CategoryTransactionRepository>((ref) {
  final api = ref.watch(categoryTransactionApiProvider);
  return CategoryTransactionRepository(api);
});

class CategoryTransactionApi {
  final ApiClient _apiClient;

  CategoryTransactionApi(this._apiClient);

  // Fetch category transactions by time range
  Future<List<Transaction>> getCategoryTransactions({
    required int budgetPk,
    required int categoryPk,
    required String startDate,
    required String endDate,
    required String aiCategory,
  }) async {
    try {
      // API 명세서에 따라 body로 파라미터를 전송
      final Map<String, dynamic> body = {
        'startDate': startDate,
        'endDate': endDate,
        'aiCategory': aiCategory,
      };


      final response = await _apiClient.post(
        '/mm/budget/detail',
        data: body,
      );

      if (response.statusCode == 200) {
        final data = response.data['data'];
        final List<dynamic> households = data['households'] ?? [];
        
        // 응답 데이터를 Transaction 모델로 변환
        return convertHouseholdToTransactions(households);
      } else {
        throw Exception('Failed to load category transactions: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to load category transactions: $e');
    }
  }
}

class CategoryTransactionRepository {
  final CategoryTransactionApi _api;

  CategoryTransactionRepository(this._api);

  Future<List<Transaction>> getCategoryTransactions({
    required int budgetPk,
    required int categoryPk,
    required String startDate,
    required String endDate,
    required String categoryName,
  }) async {

    print('🍀🍀🍀 요청정보: budgetPk=$budgetPk, categoryPk=$categoryPk, startDate=$startDate, endDate=$endDate, categoryName=$categoryName');
    
    return _api.getCategoryTransactions(
      budgetPk: budgetPk,
      categoryPk: categoryPk,
      startDate: startDate,
      endDate: endDate,
      aiCategory: categoryName,
    );
  }
}