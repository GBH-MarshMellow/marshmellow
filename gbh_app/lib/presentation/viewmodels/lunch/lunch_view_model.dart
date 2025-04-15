import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/constants/lunch_menu_data.dart';

// 점심 메뉴 상태 관리를 위한 ViewModel Provider
final lunchViewModelProvider = ChangeNotifierProvider<LunchViewModel>((ref) {
  return LunchViewModel();
});

// 점심 메뉴 상태 관리 ViewModel
class LunchViewModel extends ChangeNotifier {
  // 선택된 메뉴 목록 (중복 허용)
  final List<LunchMenu> _selectedMenus = [];
  LunchMenu? _finalSelectedMenu;

  // 선택된 메뉴 목록 getter
  List<LunchMenu> get selectedMenus => _selectedMenus;

  // 최종 선택된 메뉴 getter
  LunchMenu? get finalSelectedMenu => _finalSelectedMenu;

  // 최대 선택 가능한 메뉴 개수
  static const int maxSelectableMenus = 8;

  // 메뉴를 선택하는 메서드
  void selectMenu(LunchMenu menu) {
    // 최대 선택 가능 개수를 초과하면 선택 불가
    if (_selectedMenus.length >= maxSelectableMenus) {
      return;
    }
    
    // 메뉴를 선택 목록에 추가
    _selectedMenus.add(menu);
    notifyListeners();
  }

  // 선택된 메뉴를 취소하는 메서드 (특정 인덱스의 메뉴 제거)
  void unselectMenuAt(int index) {
    if (index >= 0 && index < _selectedMenus.length) {
      _selectedMenus.removeAt(index);
      notifyListeners();
    }
  }

  // 모든 선택을 초기화하는 메서드
  void clearAllSelections() {
    _selectedMenus.clear();
    notifyListeners();
  }
  // 최종 메뉴를 선택하는 메서드
  void selectFinalMenu(LunchMenu menu) {
    _finalSelectedMenu = menu;
    notifyListeners();
  }

  // 특정 메뉴가 최대 선택 가능 개수에 도달했는지 확인
  bool get isMaxSelected => _selectedMenus.length >= maxSelectableMenus;
}