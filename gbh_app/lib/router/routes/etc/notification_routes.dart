// lib/router/routes/notification_routes.dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/presentation/pages/etc/notifications/notifications_page.dart';

class NotificationRoutes {
  static const String root = '/notification';
  // 추가 경로가 필요하면 여기에 정의
}

List<RouteBase> notificationRoutes = [
  GoRoute(
    path: NotificationRoutes.root,
    builder: (context, state) => const NotificationsPage(),
    routes: [
      // 하위 라우트 추가 가능
    ],
  ),
];
