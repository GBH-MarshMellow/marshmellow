// core/lifecycle/app_lifecycle_manager.dart
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

// 상태 변경 처리를 위한 스트림 프로바이더 추가
final lifecycleStateProvider = StateProvider<String>((ref) => "초기화됨");

class AppLifecycleManager {
  late final AppLifecycleListener _lifecycleListener;
  final Ref ref;

  AppLifecycleManager(this.ref) {
    _init();
  }

  void _init() {
    _lifecycleListener = AppLifecycleListener(
      onStateChange: _onAppLifecycleStateChange,
    );
  }

  void _onAppLifecycleStateChange(AppLifecycleState state) {
    switch (state) {
      case AppLifecycleState.resumed:
        _handleAppResumed();
        break;
      case AppLifecycleState.paused:
        _handleAppPaused();
        break;
      case AppLifecycleState.detached:
        _handleAppDetached();
        break;
      default:
        break;
    }
  }

  void _handleAppResumed() {
    if (kDebugMode) {
      print('앱이 포그라운드로 돌아옴');
    }
    // 인증 상태 체크 등의 로직 추가 예정
    // ref.read(authProvider.notifier).checkAuthStatus();
  }

  void _handleAppPaused() {
    if (kDebugMode) {
      print('앱이 백그라운드로 전환됨');
    }
    // 백그라운드 진입시 로직 추가 예정
    // ref.read(sessionProvider.notifier).startInactivityTimer();
  }

  void _handleAppDetached() {
    if (kDebugMode) {
      print('앱이 종료됨');
    }
    // 앱 종료시 로직 추가 예정
    // ref.read(securityProvider.notifier).clearSensitiveData();
  }

  void dispose() {
    _lifecycleListener.dispose();
  }
}