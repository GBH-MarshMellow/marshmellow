// lib/router/routes/ledger_routes.dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/presentation/pages/ledger/ledger_page.dart';
import 'package:marshmellow/presentation/pages/ledger/ledger_search_page.dart';
import 'package:marshmellow/presentation/pages/ledger/ledger_analysis_page.dart';

class LedgerRoutes {
  static const String root = '/ledger';
  static const String search = 'search'; // 검색 페이지 경로
  static const String analysis = 'analysis'; // 분석 페이지 경로
  // 추가 경로가 필요하면 여기에 정의

  // 전체 경로 생성 헬퍼 메서드
  static String getSearchPath() => '$root/$search';
  static String getAnalysisPath() => '$root/$analysis';
  // 필요한 경우 추가
}

// 실제 라우트 정의
List<RouteBase> ledgerRoutes = [
  GoRoute(
    path: LedgerRoutes.root,
    builder: (context, state) => const LedgerPage(),
    routes: [
      // 하위 라우트 추가
      // 검색 페이지 라우트
      GoRoute(
        path: LedgerRoutes.search,
        builder: (context, state) => const LedgerSearchPage(),
      ),
      // 분석 페이지 라우트
      GoRoute(
        path: LedgerRoutes.analysis,
        builder: (context, state) => const LedgerAnalysisPage(),
      ),
    ],
  ),
];
