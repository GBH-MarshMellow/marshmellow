import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class EditableMemoField extends StatefulWidget {
  final String label;
  final String? initialValue;
  final Function(String) onChanged;
  final bool showDivider;
  final EdgeInsetsGeometry? padding;
  final double labelWidth; // 라벨 영역의 너비 설정
  final MainAxisAlignment rowAlignment; // 행 정렬 속성
  final bool enabled;

  const EditableMemoField({
    Key? key,
    required this.label,
    this.initialValue,
    required this.onChanged,
    this.showDivider = true,
    this.padding = const EdgeInsets.symmetric(horizontal: 10, vertical: 17),
    this.labelWidth = 100, // 기본 라벨 너비
    this.rowAlignment = MainAxisAlignment.spaceBetween, // 기본 행 정렬
    this.enabled = true,
  }) : super(key: key);

  @override
  State<EditableMemoField> createState() => _EditableMemoFieldState();
}

class _EditableMemoFieldState extends State<EditableMemoField> {
  bool _isEditing = false;
  late TextEditingController _controller;
  final FocusNode _focusNode = FocusNode();

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.initialValue ?? '');
    _focusNode.addListener(_onFocusChange);
  }

  @override
  void dispose() {
    _controller.dispose();
    _focusNode.removeListener(_onFocusChange);
    _focusNode.dispose();
    super.dispose();
  }

  void _onFocusChange() {
    if (!_focusNode.hasFocus) {
      setState(() {
        _isEditing = false;
      });
      widget.onChanged(_controller.text);
    }
  }

  void _startEditing() {
    if (!widget.enabled) return;
    setState(() {
      _isEditing = true;
    });
    // 포커스를 주면 키보드가 자동으로 올라옴
    FocusScope.of(context).requestFocus(_focusNode);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        if (_isEditing)
          // 편집 모드
          InkWell(
            child: Padding(
              padding: widget.padding!,
              child: Row(
                mainAxisAlignment: widget.rowAlignment,
                children: [
                  // 라벨 텍스트 (고정 너비)
                  SizedBox(
                    width: widget.labelWidth,
                    child: Text(
                      widget.label,
                      style: AppTextStyles.bodySmall
                          .copyWith(color: AppColors.textSecondary),
                    ),
                  ),

                  // 입력 필드 (Expanded로 남은 공간 사용)
                  Expanded(
                    child: TextField(
                      controller: _controller,
                      focusNode: _focusNode,
                      textAlign: TextAlign.right,
                      autofocus: true,
                      decoration: InputDecoration(
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.zero,
                        isDense: true,
                      ),
                      style: AppTextStyles.bodySmall.copyWith(
                        color: AppColors.textPrimary,
                      ),
                      onSubmitted: (value) {
                        setState(() {
                          _isEditing = false;
                        });
                        widget.onChanged(value);
                        _focusNode.unfocus();
                      },
                    ),
                  ),
                ],
              ),
            ),
          )
        else
          // 일반 모드 (TransactionField와 동일한 스타일)
          InkWell(
            onTap: widget.enabled ? _startEditing : null,
            child: Padding(
              padding: widget.padding!,
              child: Row(
                mainAxisAlignment: widget.rowAlignment,
                children: [
                  // 라벨 텍스트 (고정 너비)
                  SizedBox(
                    width: widget.labelWidth,
                    child: Text(
                      widget.label,
                      style: AppTextStyles.bodySmall
                          .copyWith(color: AppColors.textSecondary),
                    ),
                  ),

                  // 오른쪽 텍스트 (Expanded로 남은 공간 사용)
                  Expanded(
                    child: Align(
                      alignment: Alignment.centerRight,
                      child: Text(
                        _controller.text.isEmpty ? '' : _controller.text,
                        style: AppTextStyles.bodySmall.copyWith(
                          color: widget.enabled
                              ? AppColors.textPrimary
                              : AppColors.textSecondary,
                        ),
                        textAlign: TextAlign.right,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        if (widget.showDivider)
          const Divider(
            height: 0.5,
            color: AppColors.textLight,
          ),
      ],
    );
  }
}
