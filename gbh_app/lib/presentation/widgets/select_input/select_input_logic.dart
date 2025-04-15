import 'package:flutter/material.dart';

// 리스트 아이템을 빌드하기 위한 함수
typedef ItemBuilder<T> = Widget Function(
    BuildContext context, T item, bool isSelected);

class SelectInputController<T> {
  final TextEditingController textController;
  final ValueChanged<T>? onItemSelected;
  final String Function(T)? displayStringForItem;

  final FocusNode focusNode = FocusNode();
  bool _isFocused = false;
  T? _selectedItem;

  // 생성자
  SelectInputController({
    required this.textController,
    this.onItemSelected,
    this.displayStringForItem,
  }) {
    // 포커스 노드 리스너 설정
    focusNode.addListener(_handleFocusChange);
  }

  // 현재 입력창이 포커스되어 있는지 여부
  bool get isFocused => _isFocused;

  // 현재 선택된 아이템
  T? get selectedItem => _selectedItem;

  // 포커스 변경 이벤트 처리
  void _handleFocusChange() {
    _isFocused = focusNode.hasFocus;
  }

  // 아이템 선택 및 텍스트 필드 업데이트
  void selectItem(T item) {
    _selectedItem = item;

    // 적절한 표시 문자열로 텍스트 컨트롤러 업데이트
    if (displayStringForItem != null) {
      textController.text = displayStringForItem!(item);
    } else {
      textController.text = item.toString();
    }

    // 리스너가 있다면 알림
    if (onItemSelected != null) {
      onItemSelected!(item);
    }
  }

  // 리소스 해제
  void dispose() {
    focusNode.removeListener(_handleFocusChange);
    focusNode.dispose();
  }

  // 입력창 포커스 해제
  void clearFocus() {
    focusNode.unfocus();
  }
}
