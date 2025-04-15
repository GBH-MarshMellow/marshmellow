// lib/router/routes/my_routes.dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/presentation/pages/my/my_page.dart';
import 'package:marshmellow/presentation/pages/my/security_test_page.dart';
import 'package:marshmellow/presentation/pages/security/encryption_test_page.dart';
// lib/router/routes/my_routes.dart에서

import 'package:marshmellow/presentation/pages/testpage/datepickertest.dart';
import 'package:marshmellow/presentation/widgets/datepicker/date_picker_overlay.dart';

class MyRoutes {
  static const String root = '/my';
  static const String datepickerTest = 'datepicker-test';
  static const String securityTest = 'security-test';
  static const String encryptionTest = 'encryption-test';
  
  // 전체 경로 생성 헬퍼 메서드
  static String getDatepickerTestPath() => '$root/$datepickerTest'; // 전체경로의 형태
  static String getSecurityTestPath() => '$root/$securityTest';
  static String getEncryptionTestPath() => '$root/$encryptionTest';
}

List<RouteBase> myRoutes = [
  GoRoute(
    path: MyRoutes.root,
    builder: (context, state) => const MyPage(),
    routes: [
      GoRoute(
        path: MyRoutes.datepickerTest,
        pageBuilder: (context, state) {
          return CustomTransitionPage(
            child: DatePickerOverlay(
              child: const Datepickertest(),
            ), 
            transitionsBuilder: (context, animation, secondaryAnimation, child) {
              return FadeTransition(
                opacity: animation, 
                child: child,
              );
            }
          );
        }
      ),
      GoRoute(
        path: MyRoutes.securityTest,
        builder: (context, state) => const SecurityTestPage(),
      ),
      GoRoute(
        path: MyRoutes.encryptionTest,
        builder: (context, state) => const EncryptionTestPage(),
      ),
    ],
  ),
];