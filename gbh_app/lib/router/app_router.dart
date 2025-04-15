// lib/router/app_router.dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/utils/back_gesture/controller.dart';
import 'package:marshmellow/presentation/pages/auth/auth_check_page.dart';
// 메인 레이아웃을 관리할 위젯 import
import 'package:marshmellow/router/scaffold_with_nav_bar.dart';
// 각 탭 라우트 파일 import
import 'routes/ledger_routes.dart';
import 'routes/finance_routes.dart';
import 'routes/budget_routes.dart';
import 'routes/cookie_routes.dart';
import 'routes/my_routes.dart';
import 'routes/etc/notification_routes.dart';
import 'routes/auth_routes.dart';

/* 
  스왑뒤로가기 제스처 컨트롤을 위한
  라우터 관찰자 클래스
*/
class GoRouterObserver extends NavigatorObserver {
  final BackGestureController controller;

  GoRouterObserver(this.controller);

  @override
  void didPush(Route<dynamic> route, Route<dynamic>? previousRoute) {
    super.didPush(route, previousRoute);
    
    // 라우트 이름 또는 경로를 컨트롤러에 전달
    final String? routeName = route.settings.name;
    final String? routePath = route.settings.name ??
                              (route is PageRoute ? route.settings.arguments?.toString() : null);
    
    // 빌드 사이클 이후로 컨트롤러 업데이트 지연
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (routeName != null) {
        controller.updatePath(routeName);
      } else if (routePath != null) {
        controller.updatePath(routePath);
      }
    });
  }
}
// 라우터 생성 함수
GoRouter createRouter(BackGestureController? backgestureController) {
  // 옵저버 목록 : 컨트롤러가 제공된 경우에만 추가
  final observers = <NavigatorObserver>[];
  if (backgestureController != null) {
    observers.add(GoRouterObserver(backgestureController));
  }

  return GoRouter(
    initialLocation: '/',
    debugLogDiagnostics: true, // 개발 중에는 디버그 모드 활성화
    // 루트 경로에 대한 리다이렉트 추가
    // redirect: (context, state) {
    //   if (state.matchedLocation == '/') {
    //     return BudgetRoutes.root; 
    //   }
    //   return null;
    // },
    // 라우트 변경 감지를 위한 observer 추가
    observers: observers,
    routes: [
      // 인증 체크 페이지 (초기 라우트)
      GoRoute(
        path: '/',
        builder: (context, state) => const AuthCheckPage(),
      ),
      // 하단 네브바를 포함한 레이아웃
      ShellRoute(
        builder: (context, state, child) {
        return ScaffoldWithNavBar(child: child);
        },
        routes: [
          ...ledgerRoutes,
          ...financeRoutes,
          ...budgetRoutes,
          ...cookieRoutes,
          ...myRoutes,
          ...notificationRoutes,
        ],
      ),
      // 하단 네브바 미포함 레이아웃
      ...signupRoutes,
    ],
  );
}

// 기존 라우터 설정 : 컨트롤러 없이 사용
final goRouter = createRouter(null);