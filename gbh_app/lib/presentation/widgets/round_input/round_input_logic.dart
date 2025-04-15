import 'package:flutter/material.dart';

class RoundInputLogic {
  late FocusNode focusNode;
  late TextEditingController controller;
  bool hasText = false;
  bool isFocused = false;
  final VoidCallback? onStateChanged;
  final bool isDropdownMode;

  // 마지막 컨트롤러 값 추적을 위한 변수
  String _lastControllerValue = '';

  RoundInputLogic({
    FocusNode? externalFocusNode,
    TextEditingController? externalController,
    this.onStateChanged,
    this.isDropdownMode = false,
  }) {
    // null 안전성 확보를 위해 항상 새 인스턴스 생성
    focusNode = externalFocusNode ?? FocusNode();
    controller = externalController ?? TextEditingController();
    _lastControllerValue = controller.text;

    // 리스너 등록
    focusNode.addListener(_handleFocusChange);
    controller.addListener(_updateTextStatus);

    // 초기 상태 설정
    hasText = controller.text.isNotEmpty;
  }

  void _handleFocusChange() {
    isFocused = focusNode.hasFocus;
    onStateChanged?.call();
  }

  void _updateTextStatus() {
    // 텍스트 상태 업데이트
    final newHasText = controller.text.isNotEmpty;
    if (hasText != newHasText) {
      hasText = newHasText;
      onStateChanged?.call();
    }

    // 컨트롤러 값 변경 감지
    if (_lastControllerValue != controller.text) {
      _lastControllerValue = controller.text;
      onStateChanged?.call();
    }
  }

  void updateController(TextEditingController? newController) {
    if (newController == null) return;

    // 기존 컨트롤러 리스너 제거
    controller.removeListener(_updateTextStatus);

    // 새 컨트롤러 할당 및 리스너 등록
    controller = newController;
    controller.addListener(_updateTextStatus);

    // 상태 업데이트
    _lastControllerValue = controller.text;
    hasText = controller.text.isNotEmpty;

    onStateChanged?.call();
  }

  void clearText() {
    controller.clear();
  }

  String getDisplayText(String? hintText) {
    return hasText ? controller.text : (hintText ?? '');
  }

  TextStyle getTextStyle(TextStyle activeStyle, TextStyle hintStyle) {
    return hasText ? activeStyle : hintStyle;
  }

  void dispose() {
    // 리스너 제거
    controller.removeListener(_updateTextStatus);
    focusNode.removeListener(_handleFocusChange);

    // 외부 제공 리소스는 외부에서 관리하도록 함
  }

  // 드롭다운 탭 핸들러 (로직만 처리)
  void handleDropdownTap(VoidCallback? onDropdownTap, VoidCallback? onTap) {
    if (isDropdownMode) {
      if (onDropdownTap != null) {
        onDropdownTap();
      } else if (onTap != null) {
        onTap();
      }
    }
  }
}

// ===================== 사용 예시 =====================
/*
// RoundInput 위젯 사용 예시

// 1. 기본 사용법
RoundInput(
  onChanged: (value) {
    print('입력값: $value');
  },
)

// 2. 힌트 텍스트를 사용한 방법
RoundInput(
  hintText: '입금자명 4글자',
  onChanged: (value) {
    print('입력값: $value');
  },
)

// 3. 라벨을 사용한 방법
RoundInput(
  label: '입금자명',
  hintText: '입금자명 4글자',
  onChanged: (value) {
    print('입력값: $value');
  },
)

// 4. 컨트롤러를 사용한 방법
final TextEditingController _roundController = TextEditingController();

RoundInput(
  label: '입금자명',
  hintText: '입금자명 4글자',
  controller: _roundController,
  onChanged: (value) {
    print('입력된 값: $value');
    // 컨트롤러를 통해서도 접근 가능: _roundController.text
  },
)


// 5. 드롭다운 모드 사용 예시 (사용자가 직접 텍스트 입력 불가)
final TextEditingController _categoryController = TextEditingController();
final List<String> categories = ['식사', '카페/디저트', '쇼핑', '문화/여가', '교통'];

RoundInput(
  label: '카테고리',
  controller: _categoryController,
  hintText: '카테고리 선택',
  showDropdown: true,
  onDropdownTap: () {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (context) => Modal(
        backgroundColor: AppColors.whiteLight,
        title: '카테고리 선택',
        child: ListView.builder(
          shrinkWrap: true,
          itemCount: categories.length,
          itemBuilder: (context, index) {
            return ListTile(
              title: Text(categories[index]),
              onTap: () {
                _categoryController.text = categories[index];
                Navigator.pop(context);
              },
            );
          },
        ),
      ),
    );
  },
)

*/
