import 'package:marshmellow/data/datasources/remote/api_client.dart';
import 'package:marshmellow/data/models/budget/budget_model.dart';

class BudgetApi {
  final ApiClient _apiClient;
  
  BudgetApi(this._apiClient);

  // 전체 예산 조회
  Future<List<BudgetModel>> getAllBudgets() async {
    try {
      final response = await _apiClient.get('/mm/budget');

      if (response.statusCode == 200) {

        // 데이터 null 체크 및 안전한 접근
        final responseData = response.data;
        if (responseData == null) {
          print('예산 api 응답 데이터가 null 입니다.');
          return [];
        }

        final data = response.data['data'];

        if (data == null) {
          print('예산 api 응답의 data 필드가 null 입니다');
          return [];
        }

        final budgetList = data['budgetList'] as List;

        if (budgetList == null) {
          print('data의 budgetList 필드가 null 입니다.');
          return [];
        }
        return budgetList.map((budget) => BudgetModel.fromJson(budget)).toList();
      } else {
        throw Exception('Failed to load budgets: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to load budgets: $e');
    }
  }

  // 세부 예산 조회
  Future<List<BudgetCategoryModel>> getBudgetDetail(int budgetPk) async {
    try {
      final response = await _apiClient.get('/mm/budget/detail/$budgetPk');
      
      if (response.statusCode == 200) {
        final data = response.data['data'];
        final categoryList = data['budgetCategoryList'] as List;
        return categoryList.map((category) => BudgetCategoryModel.fromJson(category)).toList();
      } else {
        throw Exception('Failed to load budget details: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to load budget details: $e');
    }
  }

  // 세부 예산 수정
  Future<Map<String, dynamic>> updateBudgetCategory(int budgetCategoryPk, int budgetAmount) async {
    try {
      final response = await _apiClient.put(
        '/mm/budget/detail/$budgetCategoryPk',
        data: {'budgetCategoryPrice': budgetAmount}
      );
      
      if (response.statusCode == 200) {
        return response.data['data'];
      } else {
        throw Exception('Failed to update budget: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to update budget: $e');
    }
  }

  // 오늘의 예산 조회
  Future<DailyBudgetModel> getDailyBudget() async {
    try {
      final response = await _apiClient.get('/mm/budget/daily');
      
      if (response.statusCode == 200) {
        final data = response.data['data'];
        return DailyBudgetModel.fromJson(data);
      } else {
        throw Exception('Failed to load daily budget: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to load daily budget: $e');
    }
  }

  // 예산 알림 수정
  Future<Map<String, dynamic>> updateBudgetAlarm(String alarmTime) async {
    try {
      final response = await _apiClient.put(
        '/mm/budget/alarm',
        data: {'budgetAlarmTime': alarmTime}
      );
      
      if (response.statusCode == 200) {
        return response.data['data'];
      } else {
        throw Exception('Failed to update budget alarm: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to update budget alarm: $e');
    }
  }

  // 예산 생성
  Future<Map<String, dynamic>> createBudget(Map<String, dynamic> budgetData) async {
    try {
      final response = await _apiClient.post(
        '/mm/budget',
        data: budgetData,
      );

      if (response.statusCode == 200) {
        return response.data['data'];
      } else {
        throw Exception('예산 생성에 실패하였습니다: ${response.statusCode}');
      }
     } catch (e) {
      throw Exception('예산 생성에 실패! : $e');
     }
  }
}
