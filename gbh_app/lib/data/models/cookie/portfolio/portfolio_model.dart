import 'package:marshmellow/data/models/cookie/portfolio/portfolio_category_model.dart';

class PortfolioModel {
  final int portfolioPk;
  final String fileUrl;
  final String createDate;
  final String createTime;
  final String originFileName;
  final String fileName;
  final String portfolioMemo;
  final PortfolioCategoryModel portfolioCategory;

  PortfolioModel({
    required this.portfolioPk,
    required this.fileUrl,
    required this.createDate,
    required this.createTime,
    required this.originFileName,
    required this.fileName,
    required this.portfolioMemo,
    required this.portfolioCategory,
  });

  factory PortfolioModel.fromJson(Map<String, dynamic> json) {
    return PortfolioModel(
      portfolioPk: json['portfolioPk'] as int,
      fileUrl: json['fileUrl'] as String,
      createDate: json['createDate'] as String,
      createTime: json['createTime'] as String,
      originFileName: json['originFileName'] as String,
      fileName: json['fileName'] as String,
      portfolioMemo: json['portfolioMemo'] as String? ?? '',
      portfolioCategory:
          PortfolioCategoryModel.fromJson(json['portfolioCategory']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'portfolioPk': portfolioPk,
      'fileUrl': fileUrl,
      'createDate': createDate,
      'createTime': createTime,
      'originFileName': originFileName,
      'fileName': fileName,
      'portfolioMemo': portfolioMemo,
      'portfolioCategory': portfolioCategory.toJson(),
    };
  }

  // 날짜, 시간 문자열을 DateTime 객체로 변환
  DateTime get dateTime {
    try {
      final year = int.parse(createDate.substring(0, 4));
      final month = int.parse(createDate.substring(4, 6));
      final day = int.parse(createDate.substring(6, 8));

      final hour = int.parse(createTime.substring(0, 2));
      final minute = int.parse(createTime.substring(2, 4));

      return DateTime(year, month, day, hour, minute, 0);
    } catch (e) {
      return DateTime.now();
    }
  }

  // 포맷된 날짜 문자열 반환
  String get formattedDate {
    final date = dateTime;
    return '${date.year}.${date.month.toString().padLeft(2, '0')}.${date.day.toString().padLeft(2, '0')}';
  }
}

// 포트폴리오 목록 응답을 처리하기 위한 클래스
class PortfolioListResponse {
  final int code;
  final String message;
  final PortfolioListData data;

  PortfolioListResponse({
    required this.code,
    required this.message,
    required this.data,
  });

  factory PortfolioListResponse.fromJson(Map<String, dynamic> json) {
    return PortfolioListResponse(
      code: json['code'] as int,
      message: json['message'] as String,
      data: PortfolioListData.fromJson(json['data']),
    );
  }
}

class PortfolioListData {
  final List<PortfolioModel> portfolioList;

  PortfolioListData({
    required this.portfolioList,
  });

  factory PortfolioListData.fromJson(Map<String, dynamic> json) {
    return PortfolioListData(
      portfolioList: (json['portfolioList'] as List)
          .map((item) => PortfolioModel.fromJson(item))
          .toList(),
    );
  }
}
