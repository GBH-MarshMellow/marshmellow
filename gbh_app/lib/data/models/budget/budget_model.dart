import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';

/*
  전체 예산 조회 API 응답 모델
*/
class BudgetModel {
  final int budgetPk;
  final int budgetAmount;
  final String startDate;
  final String endDate;
  final List<BudgetCategoryModel> budgetCategoryList;

  BudgetModel({
    required this.budgetPk,
    required this.budgetAmount,
    required this.startDate,
    required this.endDate,
    required this.budgetCategoryList,
  });

  factory BudgetModel.fromJson(Map<String, dynamic> json) {
    return BudgetModel(
      budgetPk: json['budgetPk'],
      budgetAmount: json['budgetAmount'],
      startDate: json['startDate'],
      endDate: json['endDate'],
      budgetCategoryList: (json['budgetCategoryList'] as List)
          .map((item) => BudgetCategoryModel.fromJson(item))
          .toList(),
    );
  }

  static Map<String, Color> getCategoryColors() {
    return {
      '식비/외식': AppColors.yellowPrimary,
      '교통/자동차' : AppColors.bluePrimary,
      '여가비' : AppColors.greenLight,
      '커피/디저트' : AppColors.yellowLight,
      '쇼핑' : AppColors.pinkLight,
      '편의점/마트' : AppColors.pinkPrimary,
      '비상금' : AppColors.blueLight,
      '금융' : AppColors.blueDark,
      '고정지출' : AppColors.greyLight,
    };
  }

  // 카테고리별 색상 가져오기
  static Color getCategoryColor(String categoryName) {
    final colors = getCategoryColors();
    return colors[categoryName] ?? AppColors.background;
  }
}

/*
  예산 카테고리 모델
*/
class BudgetCategoryModel {
  final int budgetCategoryPk;
  final String budgetCategoryName;
  final int budgetCategoryPrice;
  final int? budgetExpendAmount;
  final double? budgetExpendPercent;

  BudgetCategoryModel({
    required this.budgetCategoryPk,
    required this.budgetCategoryName,
    required this.budgetCategoryPrice,
    this.budgetExpendAmount,
    this.budgetExpendPercent,
  });

  factory BudgetCategoryModel.fromJson(Map<String, dynamic> json) {
    return BudgetCategoryModel(
      budgetCategoryPk: json['budgetCategoryPk'], 
      budgetCategoryName: json['budgetCategoryName'], 
      budgetCategoryPrice: json['budgetCategoryPrice'],
      budgetExpendAmount: json['budgetExpendAmount'],
      budgetExpendPercent: json['budgetExpendPercent'],
    );
  }
  // 색상 가져오기
  Color get color {
    return BudgetModel.getCategoryColor(budgetCategoryName);
  }
}

/*
  오늘의 예산 모델
*/
class DailyBudgetModel {
  final int budgetPk;
  final int budgetAmount;
  final int remainBudgetAmount;
  final int dailyBudgetAmount;

  DailyBudgetModel({
    required this.budgetPk,
    required this.budgetAmount,
    required this.remainBudgetAmount,
    required this.dailyBudgetAmount,
  });

  factory DailyBudgetModel.fromJson(Map<String, dynamic> json) {
    return DailyBudgetModel(
      budgetPk: json['budgetPk'],
      budgetAmount: json['budgetAmount'],
      remainBudgetAmount: json['remainBudgetAmount'],
      dailyBudgetAmount: json['dailyBudgetAmount'],
    );
  }
}