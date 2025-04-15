import 'package:flutter/widgets.dart';
import 'package:flutter/scheduler.dart';

/*
  왼>오 스와이프 : 뒤로가기
*/
class BackGestureController extends ChangeNotifier {
  final Set<String> _disabledPaths = {
    // 왼>오 스와이프 뒤로가기 기능을 비활성화할 경로를 추가하세요
    '/budget/signuptest'
  };

  String _currentPath = ''; // 현재 경로
  bool _isGestureEnabled = true; // 현재 제스처 가능 여부

  // getter
  bool get isGestureEnabled => _isGestureEnabled;
  String get currentPath => _currentPath;

  // 경로 업데이트(라우트 변경시 호출)
  void updatePath(String path) {
    _currentPath = path;
    // 현재 경로가 비활성화 목록에 있는지 확인
    final newEnabled = !_disabledPaths.contains(path);
    
    // 값이 변경된 경우에만 알림
    if (_isGestureEnabled != newEnabled) {
      _isGestureEnabled = newEnabled;
      
      // 다음 프레임에서 알림 (빌드 사이클 이후로 지연)
      SchedulerBinding.instance.addPostFrameCallback((_) {
        notifyListeners();
      });
    }
  }

  // 수동으로 제스처 활성화/비활성화
  void setGestureEnabled(bool enabled) {
    _isGestureEnabled = enabled;
    notifyListeners();
  }
}