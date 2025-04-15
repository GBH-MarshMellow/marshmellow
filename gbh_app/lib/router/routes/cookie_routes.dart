import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/presentation/pages/cookie/cookie_page.dart';
import 'package:marshmellow/presentation/pages/cookie/lunch_page/lunch_page.dart';
import 'package:marshmellow/presentation/pages/cookie/lunch_page/lunch_run_page.dart';
import 'package:marshmellow/presentation/pages/cookie/lunch_page/lunch_tutorial_page.dart';
import 'package:marshmellow/presentation/pages/cookie/quit_page/quit_page.dart';
import 'package:marshmellow/presentation/pages/cookie/portfolio_page/portfolio_page.dart';
import 'package:marshmellow/presentation/pages/cookie/quit_page/quit_info_page.dart';
import 'package:marshmellow/presentation/pages/cookie/portfolio_page/portfolio_category_detail_page.dart';

class CookieRoutes {
  static const String root = '/cookie';
  // 추가 경로가 필요하면 여기에 정의
  static const String lunch = 'lunch';
  static const String quit = 'quit';
  static const String portfolio = 'portfolio';
  static const String info = 'info';
  static const String tutorial = 'tutorial';
  static const String run = 'run';
  static const String portfolioCategory = 'portfolio-category';

  static String getLunchPath() => '$root/$lunch';
  static String getQuitPath() => '$root/$quit';
  static String getPortfolioPath() => '$root/$portfolio';

  static String getLunchTutorialPath() => '$root/$lunch/$tutorial';
  static String getLunchRunPath() => '$root/$lunch/$run';

  static String getQuitInfoPath() => '$root/$quit/$info';
  static String getPortfolioCategoryDetailPath(int categoryPk) =>
      '$root/$portfolio/$portfolioCategory/$categoryPk';
}

List<RouteBase> cookieRoutes = [
  GoRoute(
    path: CookieRoutes.root,
    builder: (context, state) => const CookiePage(),
    routes: [
      GoRoute(
        path: 'lunch',
        builder: (context, state) => const LunchPage(),
        routes: [
          GoRoute(
            path: CookieRoutes.tutorial,
            builder: (context, state) => const LunchTutorialPage(),
          ),
          GoRoute(
            path: CookieRoutes.run,
            builder: (context, state) => const LunchRunPage(),
          ),
        ],
      ),
      GoRoute(
        path: 'quit',
        builder: (context, state) => const QuitPage(),
      ),
      GoRoute(
        path: 'portfolio',
        builder: (context, state) => const PortfolioPage(),
        routes: [
          GoRoute(
            path: 'portfolio-category/:categoryPk',
            builder: (context, state) {
              final categoryPk = int.parse(state.pathParameters['categoryPk']!);
              return PortfolioDetailPage(categoryId: categoryPk);
            },
          ),
        ],
      ),
      GoRoute(
        path: 'quit/info',
        builder: (context, state) => const QuitInfoPage(),
      ),
    ],
  ),
];
