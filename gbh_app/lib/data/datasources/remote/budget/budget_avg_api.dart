import 'package:marshmellow/data/datasources/remote/api_client.dart';

/*
  지출 평균 (월급 대비) 조회 API
  : 예산 유형 조회 API 의 body
*/
class BudgetAvgApi {
  final ApiClient _apiClient;

  BudgetAvgApi(this._apiClient);

  Future<Map<String, dynamic>> getBudgetAverage() async {
    try {
      final response = await _apiClient.get('/household/ai-avg');

      if (response.statusCode == 200) {
        return response.data['data'];
      } else {
        throw Exception('월급 대비 지출 평균 조회에 실패했습니다: ${response.statusCode}');
      }
      
    } catch (e) {
      throw Exception('월급 대비 지출 평균 조회에 실패했습니다: $e');
    }
  }
}