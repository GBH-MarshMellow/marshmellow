import 'package:flutter/material.dart';

class TextInputLogic {
  late FocusNode focusNode;
  TextEditingController? controller;
  bool hasText = false;
  bool isFocused = false;
  final VoidCallback? onStateChanged;

  TextInputLogic({
    FocusNode? externalFocusNode,
    TextEditingController? externalController,
    this.onStateChanged,
  }) {
    focusNode = externalFocusNode ?? FocusNode();
    controller = externalController;

    focusNode.addListener(_handleFocusChange);
    if (controller != null) {
      controller!.addListener(_updateTextStatus);
    }
  }

  void _handleFocusChange() {
    isFocused = focusNode.hasFocus;
    onStateChanged?.call();
  }

  void _updateTextStatus() {
    final newHasText = controller?.text.isNotEmpty ?? false;
    if (hasText != newHasText) {
      hasText = newHasText;
      onStateChanged?.call();
    }
  }

  void updateController(TextEditingController? newController) {
    if (controller != null) {
      controller!.removeListener(_updateTextStatus);
    }
    controller = newController;
    if (controller != null) {
      controller!.addListener(_updateTextStatus);
      hasText = controller!.text.isNotEmpty;
    }
  }

  void clearText() {
    controller?.clear();
  }

  void dispose() {
    controller?.removeListener(_updateTextStatus);
    focusNode.removeListener(_handleFocusChange);
    focusNode.dispose();
  }
}
