import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/presentation/widgets/bottom_navbar/bottom_navbar.dart'; // ui
import 'package:marshmellow/di/providers/modal_provider.dart';

// 라우트 클래스들 import
import 'routes/ledger_routes.dart';
import 'routes/finance_routes.dart';
import 'routes/budget_routes.dart';
import 'routes/cookie_routes.dart';
import 'routes/my_routes.dart';

class ScaffoldWithNavBar extends ConsumerWidget {
  final Widget child;

  const ScaffoldWithNavBar({
    Key? key,
    required this.child,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // 모달 상태 읽기
    final isModalVisible = ref.watch(modalProvider);

    // 현재 라우트 경로를 기반으로 선택된 탭 인덱스 계산
    final String location = GoRouterState.of(context).uri.path;
    int currentIndex = 0;

    if (location.startsWith(LedgerRoutes.root)) {
      currentIndex = 0;
    } else if (location.startsWith(FinanceRoutes.root)) {
      currentIndex = 1;
    } else if (location.startsWith(BudgetRoutes.root)) {
      currentIndex = 2;
    } else if (location.startsWith(CookieRoutes.root)) {
      currentIndex = 3;
    } else if (location.startsWith(MyRoutes.root)) {
      currentIndex = 4;
    }

    // 아이콘 경로 및 라벨 설정
    final List<String> iconPaths = const [
      'assets/icons/nav/ledger_bk.svg',
      'assets/icons/nav/finance_bk.svg',
      'assets/icons/nav/budget_bk.svg',
      'assets/icons/nav/cookie_bk.svg',
      'assets/icons/nav/user_bk.svg',
    ];

    final List<String> labels = const [
      '가계',
      '자산',
      '예산',
      '쿠키',
      '마이',
    ];

    return Scaffold(
      body: child, // 현재 선택된 페이지
      bottomNavigationBar: isModalVisible // 모달이 표시 중이면 네비게이션 바를 표시하지 않음음
          ? null
          : CustomBottomNavBar(
              selectedIndex: currentIndex,
              onTap: (index) {
                // 새 탭으로 네비게이션
                switch (index) {
                  case 0:
                    context.go(LedgerRoutes.root);
                    break;
                  case 1:
                    context.go(FinanceRoutes.root);
                    break;
                  case 2:
                    context.go(BudgetRoutes.root);
                    break;
                  case 3:
                    context.go(CookieRoutes.root);
                    break;
                  case 4:
                    context.go(MyRoutes.root);
                    break;
                }
              },
              iconPaths: iconPaths,
              labels: labels,
            ),
    );
  }
}
