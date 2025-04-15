// lib/router/routes/budget_routes.dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart'; 
import 'package:marshmellow/data/models/budget/budget_model.dart';
import 'package:marshmellow/data/models/wishlist/wishlist_model.dart';
import 'package:marshmellow/presentation/pages/budget/budget_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_detail/budget_category_detail_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_detail/category_expense_list_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_salary/budget_creation_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_salary/budget_type_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_salary/budget_type_selection_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_salary/salary_celebrate_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/salary_to_wish/wish_complete_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/salary_to_wish/wish_selection_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/salary_to_wish/wish_setup_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/wish/wishlist_creation_page.dart';
import 'package:marshmellow/app.dart';

/*
  예산 routes
*/
class BudgetRoutes {
  // 예산 관련 경로 정의
  static const String root = '/budget';
  static const String budgetdetail = 'detail/:budgetPk';
  static const String budgetcategoryexpense = 'category/expenses/:categoryPk';
  static const String budgetevent = 'event';
  
  // 위시 관련 경로 정의
  static const String wishlistcreate = 'wishlist/create';

  // 예산 경로 생성 헬퍼 메서드
  static String getBudgetDetailPath() => '$root/$budgetdetail';
  static String getBudgetCategoryExpensePath() => '$root/$budgetcategoryexpense';
  static String getBudgetEventPath() => '$root/$budgetevent';
  
  // 위시 경로 생성 헬퍼 메서드
  static String getWishlistCreatePath() => '$root/$wishlistcreate';
}

List<RouteBase> budgetRoutes = [
  GoRoute(
    path: BudgetRoutes.root,
    builder: (context, state) => const BudgetPage(),
    routes: [
      // 예산 상세 페이지 : 모든 카테고리
      GoRoute(
        path: BudgetRoutes.budgetdetail,
        builder: (context, state) {
          final budgetPk = int.parse(state.pathParameters['budgetPk']!);
          return BudgetCategoryDetailPage(budgetPk: budgetPk);
        },
      ),

      // 예산 카테고리별 지출 상세 페이지
      GoRoute(
        path: BudgetRoutes.budgetcategoryexpense,
        builder: (context, state) {
          final categoryPk = int.parse(state.pathParameters['categoryPk']!);
      
          // Extra data contains the category model and budgetPk
          final extra = state.extra as Map<String, dynamic>;
          final category = extra['category'] as BudgetCategoryModel;
          final budgetPk = extra['budgetPk'] as int;
          return CategoryExpensePage(categoryPk: categoryPk, category: category, budgetPk: budgetPk);
        },
      ),

      // 위시리스트 생성 페이지
      GoRoute(
        path: BudgetRoutes.wishlistcreate,
        builder: (context, state) {
          // Consumer 위젯으로 Riverpod 상태 접근
          return Consumer(
            builder: (context, ref, child) {
              // 공유된 URL 접근
              final sharedUrl = ref.read(sharedUrlProvider);
              
              // URL 사용 후 상태 초기화
              if (sharedUrl != null) {
                Future.microtask(() {
                  ref.read(sharedUrlProvider.notifier).state = null;
                });
              }
              return WishlistCreationPage(sharedUrl: sharedUrl);
            },
          );
        },
      ),

    ],
  ),
];