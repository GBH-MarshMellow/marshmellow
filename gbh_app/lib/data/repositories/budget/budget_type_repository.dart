import 'package:marshmellow/data/datasources/remote/budget/budget_avg_api.dart';
import 'package:marshmellow/data/datasources/remote/budget/budget_type_api.dart';
import 'package:marshmellow/data/models/budget/budget_type_model.dart';

class BudgetTypeRepository {
  final BudgetTypeApi _budgetTypeApi;
  final BudgetAvgApi _budgetAvgApi;

  BudgetTypeRepository(this._budgetTypeApi, this._budgetAvgApi);

  // 예산 유형 분석
  Future<BudgetTypeAnalysisResponse> analyzeBudgetType(BudgetTypeAnalysisRequest request) async {
    return await _budgetTypeApi.analyzeBudgetType(request);
  }

  // 월급 대비 지출 평균 조회 후 예산 유형 분석 요청
  Future<BudgetTypeAnalysisResponse> anaylyzeWithApiData() async {
    try {
      // 월급 대비 지출 평균 조회
      final avgData = await _budgetAvgApi.getBudgetAverage();

      // API 응답을 BudgetTypeAnlysisRequest로 반환
      final request = BudgetTypeAnalysisRequest(
        salary: avgData['salary'] ?? 0, 
        fixedExpense: avgData['fixedAvg'] ?? 0, 
        foodExpense: avgData['foodAvg'] ?? 0, 
        transportationExpense: avgData['trafficAvg'] ?? 0, 
        marketExpense: avgData['martAvg'] ?? 0, 
        financialExpense: avgData['bankAvg'] ?? 0, 
        leisureExpense: avgData['leisureAvg'] ?? 0, 
        coffeeExpense: avgData['coffeeAvg'] ?? 0, 
        shoppingExpense: avgData['shoppingAvg'] ?? 0, 
        emergencyExpense: avgData['emergencyAvg'] ?? 0
      );

      // 예산 유형 분석 api 호출
      return await _budgetTypeApi.analyzeBudgetType(request);
    } catch (e) {
      throw Exception('예산 유형 분석 정보를 불러오는데 실패했습니다: $e');
    }
  }

  // 예산 유형 선택 저장
  Future<bool> saveBudgetTypeSelection(String selectedType) async {
    // 사용자가 선택한 예산 유형을 저장하는 것을 여기에 작성해!!!
    return true;
  }
}