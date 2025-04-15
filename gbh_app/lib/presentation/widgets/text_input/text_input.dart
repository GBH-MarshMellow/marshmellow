import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/text_input/text_input_logic.dart';

class TextInput extends StatefulWidget {
  final String label;
  final bool readOnly;
  final ValueChanged<String>? onChanged;
  final TextEditingController? controller; // 텍스트 인풋 컨트롤러
  final String? errorText; // 에러 텍스트
  final VoidCallback? onTap; // 텍스트 인풋 탭 이벤트
  final FocusNode? focusNode; // 포커스 노드
  final double? width; // 텍스트 인풋 너비

  const TextInput({
    super.key,
    required this.label,
    this.onChanged,
    this.readOnly = false,
    this.controller,
    this.errorText,
    this.onTap,
    this.focusNode,
    this.width,
  });

  @override
  State<TextInput> createState() => _TextInputState();
}

class _TextInputState extends State<TextInput> {
  late TextInputLogic _textInputLogic;

  @override
  void initState() {
    super.initState();
    _textInputLogic = TextInputLogic(
      externalFocusNode: widget.focusNode,
      externalController: widget.controller,
      onStateChanged: () => setState(() {}),
    );
  }

  @override
  void didUpdateWidget(TextInput oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.focusNode != oldWidget.focusNode) {
      _textInputLogic.dispose();
      _textInputLogic = TextInputLogic(
        externalFocusNode: widget.focusNode,
        externalController: widget.controller,
        onStateChanged: () => setState(() {}),
      );
    }
    if (widget.controller != oldWidget.controller) {
      _textInputLogic.updateController(widget.controller);
    }
  }

  @override
  void dispose() {
    _textInputLogic.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final Color borderColor = widget.errorText != null
        ? AppColors.warnning
        : _textInputLogic.isFocused
            ? AppColors.textPrimary
            : AppColors.textSecondary;

    final Color textColor = _textInputLogic.isFocused
        ? AppColors.textPrimary
        : AppColors.textSecondary;

    final double screenWidth = MediaQuery.of(context).size.width;

    return Center(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: widget.width ?? screenWidth * 0.9,
            height: 60,
            padding: const EdgeInsets.symmetric(horizontal: 15),
            decoration: BoxDecoration(
              border: Border.all(color: borderColor),
              borderRadius: BorderRadius.circular(5),
              color: AppColors.whiteLight,
            ),
            child: Stack(
              children: [
                TextField(
                  controller: _textInputLogic.controller,
                  focusNode: _textInputLogic.focusNode,
                  onChanged: (value) {
                    if (widget.onChanged != null) {
                      widget.onChanged!(value);
                    }
                  },
                  readOnly: widget.readOnly,
                  onTap: widget.onTap,
                  cursorColor: textColor,
                  decoration: InputDecoration(
                    labelText: widget.label,
                    labelStyle: AppTextStyles.bodySmall.copyWith(
                      color: textColor,
                    ),
                    border: InputBorder.none,
                    focusedBorder: InputBorder.none,
                    enabledBorder: InputBorder.none,
                    errorBorder: InputBorder.none,
                    disabledBorder: InputBorder.none,
                    contentPadding: const EdgeInsets.only(
                      top: 16,
                      right: 40,
                    ),
                  ),
                  style: AppTextStyles.bodySmall.copyWith(
                    color: textColor,
                  ),
                ),
                if (_textInputLogic.isFocused && _textInputLogic.hasText)
                  Positioned(
                    right: 0,
                    bottom: 10,
                    child: GestureDetector(
                      onTap: _textInputLogic.clearText,
                      child: Container(
                        width: 24,
                        height: 24,
                        decoration: const BoxDecoration(
                          shape: BoxShape.circle,
                          color: Colors.transparent,
                        ),
                        child: Center(
                          child: Icon(
                            Icons.cancel_outlined,
                            size: 15,
                            color: AppColors.textPrimary,
                          ),
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          if (widget.errorText != null)
            Container(
              width: widget.width ?? screenWidth * 0.9,
              padding: const EdgeInsets.only(left: 16, top: 4),
              margin: const EdgeInsets.only(bottom: 23),
              child: Text(
                widget.errorText!,
                style: AppTextStyles.bodySmall.copyWith(
                  fontWeight: FontWeight.w300,
                  color: AppColors.warnning,
                ),
              ),
            ),
          if (widget.errorText == null) const SizedBox(height: 23),
        ],
      ),
    );
  }
}

// ===================== 사용 예시 =====================
/*
// TextInput 위젯 사용 예시

// 1. 기본 사용법
TextInput(
  label: '이름',
  onChanged: (value) {
    print('이름: $value');
  },
)

// 2. 컨트롤러를 사용한 방법
final TextEditingController _nameController = TextEditingController();

TextInput(
  label: '이름',
  controller: _nameController,
  onChanged: (value) {
    print('입력된 이름: $value');
    // 컨트롤러를 통해서도 접근 가능: _nameController.text
  },
)

// 3. 유효성 검증 사용 예시
String? _emailError;

void _validateEmail(String email) {
  final emailRegex = RegExp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$');
  if (email.isEmpty) {
    setState(() => _emailError = null);
  } else if (!emailRegex.hasMatch(email)) {
    setState(() => _emailError = '유효한 이메일 주소를 입력해주세요');
  } else {
    setState(() => _emailError = null);
  }
}

TextInput(
  label: '이메일',
  errorText: _emailError,
  onChanged: (value) {
    _validateEmail(value);
    print('이메일: $value');
  },
)

// 4. 읽기 전용 및 탭 이벤트 사용 예시
TextInput(
  label: '생년월일',
  readOnly: true, // 직접 입력 불가능, 탭으로만 입력 가능
  controller: _birthController,
  onTap: () {
    // 날짜 선택 다이얼로그 표시
    showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime(1900),
      lastDate: DateTime.now(),
    ).then((selectedDate) {
      if (selectedDate != null) {
        _birthController.text = '${selectedDate.year}-${selectedDate.month}-${selectedDate.day}';
      }
    });
  },
)
*/
