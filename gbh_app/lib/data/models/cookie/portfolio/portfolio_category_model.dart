class PortfolioCategoryModel {
  final int portfolioCategoryPk;
  final String portfolioCategoryName;
  final String portfolioCategoryMemo;

  PortfolioCategoryModel({
    required this.portfolioCategoryPk,
    required this.portfolioCategoryName,
    required this.portfolioCategoryMemo,
  });

  factory PortfolioCategoryModel.fromJson(Map<String, dynamic> json) {
    return PortfolioCategoryModel(
      portfolioCategoryPk: json['portfolioCategoryPk'] as int,
      portfolioCategoryName: json['portfolioCategoryName'] as String,
      portfolioCategoryMemo: json['portfolioCategoryMemo'] as String? ?? '',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'portfolioCategoryPk': portfolioCategoryPk,
      'portfolioCategoryName': portfolioCategoryName,
      'portfolioCategoryMemo': portfolioCategoryMemo,
    };
  }
}

// 카테고리 목록 응답을 처리하기 위한 클래스
class PortfolioCategoryListResponse {
  final int code;
  final String message;
  final PortfolioCategoryListData data;

  PortfolioCategoryListResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory PortfolioCategoryListResponse.fromJson(Map<String, dynamic> json) {
    return PortfolioCategoryListResponse(
      code: json['code'] as int,
      message: json['message'] as String,
      data: PortfolioCategoryListData.fromJson(json['data']),
    );
  }
}

class PortfolioCategoryListData {
  final List<PortfolioCategoryModel> portfolioCategoryList;

  PortfolioCategoryListData({
    required this.portfolioCategoryList,
  });

  factory PortfolioCategoryListData.fromJson(Map<String, dynamic> json) {
    return PortfolioCategoryListData(
      portfolioCategoryList: (json['portfolioCategoryList'] as List)
          .map((item) => PortfolioCategoryModel.fromJson(item))
          .toList(),
    );
  }
}