import 'package:marshmellow/data/datasources/remote/api_client.dart';
import 'package:marshmellow/data/models/cookie/quit/quit_model.dart';
import 'dart:convert';

class QuitApi {
  final ApiClient _apiClient;

  QuitApi(this._apiClient);

  // 퇴사 망상 - 한달 평균 지출 조회
  Future<AverageSpendingResponse> getAverageSpending() async {
    try {
      final response = await _apiClient.get('/delusion/average');

      // 응답이 문자열인 경우 JSON으로 파싱
      if (response.data is String) {
        try {
          final Map<String, dynamic> jsonData = json.decode(response.data);
          // 전체 응답 내용 출력
          print('전체 응답: ${response.toString()}');
          print('응답 상태 코드: ${response.statusCode}');
          print('응답 데이터 타입: ${response.data.runtimeType}');
          print('응답 데이터: ${response.data}');
          return AverageSpendingResponse.fromJson(jsonData);
        } catch (e) {
          print('JSON 파싱 에러: $e');
          print('원본 응답: ${response.data}');
          throw Exception('응답을 JSON으로 파싱할 수 없습니다');
        }
      }

      // 이미 Map인 경우
      return AverageSpendingResponse.fromJson(response.data);
    } catch (e) {
      print('API 호출 에러: $e');
      throw Exception('평균 지출 데이터를 불러오는데 실패했습니다: $e');
    }
  }

  // 퇴사 망상 - 사용 가능 금액 조회
  Future<DelusionResponse> getAvailableAmount() async {
    try {
      final response = await _apiClient.get('/delusion');
      
      if (response.data is String) {
        try {
          final Map<String, dynamic> jsonData = json.decode(response.data);
          return DelusionResponse.fromJson(jsonData);
        } catch (e) {
          throw Exception('응답을 JSON으로 파싱할 수 없습니다: $e');
        }
      }
      
      return DelusionResponse.fromJson(response.data);
    } catch (e) {
      throw Exception('사용 가능 금액 데이터를 불러오는데 실패했습니다: $e');
    }
  }
}
