import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class WishlistInput extends StatefulWidget {
  final TextEditingController? controller;
  final FocusNode? focusNode;
  final String? label;
  final String? hintText;
  final String? errorText;
  final Widget? prefixIcon;
  final Widget? suffix;
  final TextInputType? keyboardType;
  final List<TextInputFormatter>? inputFormatters;
  final ValueChanged<String>? onChanged;
  final FormFieldValidator<String>? validator;
  final bool enabled;
  final String? initialValue;
  final VoidCallback? onTap;

  const WishlistInput({
    Key? key,
    this.controller,
    this.focusNode,
    this.label,
    this.hintText,
    this.errorText,
    this.prefixIcon,
    this.suffix,
    this.keyboardType,
    this.inputFormatters,
    this.onChanged,
    this.validator,
    this.enabled = true,
    this.initialValue,
    this.onTap,
  }) : super(key: key);

  @override
  State<WishlistInput> createState() => _WishlistInputState();
}

class _WishlistInputState extends State<WishlistInput> {
  late TextEditingController _controller;
  late FocusNode _focusNode;
  bool _hasFocus = false;
  String? _errorText;

  @override
  void initState() {
    super.initState();
    _controller = widget.controller ?? TextEditingController(text: widget.initialValue);
    _focusNode = widget.focusNode ?? FocusNode();
    _focusNode.addListener(_handleFocusChange);
  }

  @override
  void didUpdateWidget(WishlistInput oldWidget) {
    super.didUpdateWidget(oldWidget);
    // 컨트롤러가 변경된 경우 업데이트
    if (widget.controller != oldWidget.controller && widget.controller != null) {
      _controller = widget.controller!;
    }
    // 포커스 노드가 변경된 경우 리스너 업데이트
    if (widget.focusNode != oldWidget.focusNode) {
      oldWidget.focusNode?.removeListener(_handleFocusChange);
      _focusNode = widget.focusNode ?? _focusNode;
      _focusNode.addListener(_handleFocusChange);
    }
    // 초기값이 변경된 경우 (컨트롤러가 없을 때)
    if (widget.initialValue != oldWidget.initialValue && widget.controller == null) {
      _controller.text = widget.initialValue ?? '';
    }
  }

  void _handleFocusChange() {
    if (_focusNode.hasFocus != _hasFocus) {
      setState(() {
        _hasFocus = _focusNode.hasFocus;
      });
      
      // 포커스를 잃었을 때 유효성 검사
      if (!_hasFocus && widget.validator != null) {
        _validateInput();
      }
    }
  }
  
  // 입력값 유효성 검사
  void _validateInput() {
    if (widget.validator != null) {
      setState(() {
        _errorText = widget.validator!(_controller.text);
      });
    }
  }

  @override
  void dispose() {
    // 위젯에서 제공하지 않은 컨트롤러와 포커스 노드만 해제
    if (widget.controller == null) {
      _controller.dispose();
    }
    if (widget.focusNode == null) {
      _focusNode.removeListener(_handleFocusChange);
      _focusNode.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 입력 필드의 상태에 따른 스타일 및 동작 결정
    final bool isEnabled = widget.enabled;
    final Color borderColor = _hasFocus 
      ? AppColors.bluePrimary
      : widget.errorText != null || _errorText != null 
        ? AppColors.warnning 
        : AppColors.backgroundBlack;
    
    // 에러 메시지 결정 (위젯에서 직접 제공한 것 또는 유효성 검사 결과)
    final String? displayErrorText = widget.errorText ?? _errorText;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // 입력 필드
        GestureDetector(
          onTap: isEnabled ? widget.onTap : null,
          child: Container(
            height: 50,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(30),
              border: Border.all(
                color: borderColor,
                width: 1.0,
              ),
              color: isEnabled ? Colors.white : Colors.grey[100],
            ),
            child: Row(
              children: [
                // 접두 아이콘 또는 라벨
                if (widget.prefixIcon != null || widget.label != null)
                  Padding(
                    padding: const EdgeInsets.only(left: 16.0, right: 8.0),
                    child: widget.prefixIcon ?? 
                          Text(widget.label ?? '',
                            style: AppTextStyles.bodyExtraSmall.copyWith(
                              color: AppColors.textSecondary,
                            ),
                          ),
                  ),
                
                // 입력 필드
                Expanded(
                  child: TextField(
                    controller: _controller,
                    focusNode: _focusNode,
                    keyboardType: widget.keyboardType,
                    inputFormatters: widget.inputFormatters,
                    onChanged: (value) {
                      if (widget.onChanged != null) {
                        widget.onChanged!(value);
                      }
                      // 입력 중에도 유효성 검사 실행
                      if (widget.validator != null) {
                        _validateInput();
                      }
                    },
                    enabled: isEnabled,
                    decoration: InputDecoration(
                      hintText: widget.hintText,
                      hintStyle: AppTextStyles.bodySmall.copyWith(
                        color: AppColors.textSecondary.withOpacity(0.5),
                      ),
                      border: InputBorder.none,
                      contentPadding: const EdgeInsets.symmetric(horizontal: 16.0),
                    ),
                    style: AppTextStyles.bodyMedium,
                  ),
                ),
                
                // 접미사 위젯
                if (widget.suffix != null)
                  Padding(
                    padding: const EdgeInsets.only(right: 16.0),
                    child: widget.suffix,
                  ),
              ],
            ),
          ),
        ),
        
        // 에러 메시지
        if (displayErrorText != null)
          Padding(
            padding: const EdgeInsets.only(left: 16.0, top: 4.0),
            child: Text(
              displayErrorText,
              style: AppTextStyles.bodySmall.copyWith(
                color: AppColors.warnning,
              ),
            ),
          ),
      ],
    );
  }
}