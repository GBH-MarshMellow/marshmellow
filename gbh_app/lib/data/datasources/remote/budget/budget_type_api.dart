import 'package:marshmellow/data/datasources/remote/api_client.dart';
import 'package:marshmellow/data/models/budget/budget_type_model.dart';

class BudgetTypeApi {
  final ApiClient _apiClient;

  BudgetTypeApi(this._apiClient);

  // 예산 유형별 분석 api 호출
  Future<BudgetTypeAnalysisResponse> analyzeBudgetType(BudgetTypeAnalysisRequest request) async {
    try {
      final response = await _apiClient.post(
        '/mm/ai/type',
        data: request.toJson(),
      );

      if (response.statusCode == 200) {
        final data = response.data;
        print('🔍budget_type_api.dart 파싱할 데이터: $data');
        return BudgetTypeAnalysisResponse.fromJson(data);
      } else {
        throw Exception('Failed to load budget type analysis data: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to load budget type analysis data: $e');
    }
  }
}