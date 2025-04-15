class AverageSpendingResponse {
  final int code;
  final String message;
  final AverageSpendingData data;

  AverageSpendingResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory AverageSpendingResponse.fromJson(Map<String, dynamic> json) {
    return AverageSpendingResponse(
      code: json['code'] as int,
      message: json['message'] as String,
      data: AverageSpendingData.fromJson(json['data']),
    );
  }
}

class AverageSpendingData {
  final Map<String, int> monthlySpendingMap;
  final int averageMonthlySpending;

  AverageSpendingData({
    required this.monthlySpendingMap,
    required this.averageMonthlySpending,
  });

  factory AverageSpendingData.fromJson(Map<String, dynamic> json) {
    final Map<String, int> monthlyMap = {};
    final dynamic rawMap = json['monthlySpendingMap'];
    
    if (rawMap is Map) {
      rawMap.forEach((key, value) {
        monthlyMap[key.toString()] = value is int ? value : int.parse(value.toString());
      });
    }

    return AverageSpendingData(
      monthlySpendingMap: monthlyMap,
      averageMonthlySpending: json['averageMonthlySpending'] as int,
    );
  }
}



class DelusionResponse {
  final int code;
  final String message;
  final DelusionData data;

  DelusionResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory DelusionResponse.fromJson(Map<String, dynamic> json) {
    // 응답이 배열이면 빈 데이터로 처리
    if (json['data'] is List) {
      return DelusionResponse(
        code: json['code'] as int,
        message: json['message'] as String,
        data: DelusionData(availableAmount: 0),
      );
    }
    
    return DelusionResponse(
      code: json['code'] as int,
      message: json['message'] as String,
      data: DelusionData.fromJson(json['data']),
    );
  }
}

class DelusionData {
  final int availableAmount;

  DelusionData({
    required this.availableAmount,
  });

  factory DelusionData.fromJson(Map<String, dynamic> json) {
    final dynamic amount = json['availableAmount'];
    // String이나 다른 형태로 올 수 있으므로 변환 처리
    final int availableAmount = amount is int 
        ? amount 
        : int.parse(amount.toString());
    
    return DelusionData(
      availableAmount: availableAmount,
    );
  }
}