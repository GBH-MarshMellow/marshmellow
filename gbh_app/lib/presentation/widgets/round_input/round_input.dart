import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/presentation/widgets/round_input/round_input_logic.dart';
import 'package:marshmellow/presentation/viewmodels/modal/modal_layout.dart';

class RoundInput<T> extends StatefulWidget {
  final ValueChanged<String>? onChanged;
  final ValueChanged<T>? onItemSelected;
  final VoidCallback? onTap;
  final TextEditingController? controller;
  final FocusNode? focusNode;
  final String? errorText;
  final double? height;
  final double? width;
  final String? hintText;
  final String? label;
  final double? labelWidth;
  final bool showDropdown;
  final List<T>? items;
  final String Function(T)? displayStringForItem;
  final String? modalTitle;
  final bool showDividers;
  final bool showTitleDivider;
  final ValueChanged<String>? onSubmitted;

  const RoundInput({
    super.key,
    this.onChanged,
    this.onTap,
    this.controller,
    this.focusNode,
    this.errorText,
    this.width,
    this.height,
    this.hintText,
    this.label,
    this.labelWidth,
    this.showDropdown = false,
    this.items,
    this.onItemSelected,
    this.displayStringForItem,
    this.modalTitle,
    this.showDividers = true,
    this.showTitleDivider = false,
    this.onSubmitted,
  });

  @override
  State<RoundInput<T>> createState() => _RoundInputState<T>();
}

class _RoundInputState<T> extends State<RoundInput<T>> {
  late RoundInputLogic _inputLogic;
  T? _selectedItem;

  @override
  void initState() {
    super.initState();
    _initializeLogic();
  }

  void _initializeLogic() {
    _inputLogic = RoundInputLogic(
      externalFocusNode: widget.focusNode,
      externalController: widget.controller,
      isDropdownMode: widget.showDropdown,
      onStateChanged: () {
        if (mounted) {
          setState(() {});
        }
      },
    );
  }

  @override
  void didUpdateWidget(RoundInput<T> oldWidget) {
    super.didUpdateWidget(oldWidget);

    // 중요한 속성이 변경되면 로직을 재초기화
    if (widget.focusNode != oldWidget.focusNode ||
        widget.showDropdown != oldWidget.showDropdown) {
      _inputLogic.dispose();
      _initializeLogic();
    }

    // controller만 변경된 경우 컨트롤러만 업데이트
    else if (widget.controller != oldWidget.controller) {
      _inputLogic.updateController(widget.controller);
    }
  }

  @override
  void dispose() {
    _inputLogic.dispose();
    super.dispose();
  }

  void _showBottomSheet() {
    if (widget.items == null || widget.items!.isEmpty) return;

    WidgetsBinding.instance.addPostFrameCallback((_) {
      _inputLogic.focusNode.unfocus();
    });

    ModalLayout.showSelectionModal<T>(
      context: context,
      items: widget.items!,
      displayStringForItem: widget.displayStringForItem ?? _getDisplayText,
      onItemSelected: (item) {
        setState(() {
          _selectedItem = item;
          if (widget.controller != null) {
            widget.controller!.text = _getDisplayText(item);
          }
        });
        if (widget.onItemSelected != null) {
          widget.onItemSelected!(item);
        }
      },
      modalTitle: widget.modalTitle,
      showDividers: widget.showDividers,
      showTitleDivider: widget.showTitleDivider,
      selectedItem: _selectedItem,
    );
  }

  String _getDisplayText(T item) {
    if (widget.displayStringForItem != null) {
      return widget.displayStringForItem!(item);
    }
    return item.toString();
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildInputContainer(context),
          _buildErrorText(context),
        ],
      ),
    );
  }

  // 입력 컨테이너 위젯
  Widget _buildInputContainer(BuildContext context) {
    final double screenWidth = MediaQuery.of(context).size.width;

    // 포커스 상태에 따른 테두리 색상 결정
    final borderColor =
        _inputLogic.isFocused ? AppColors.textPrimary : AppColors.textLight;

    // 입력 상태 확인 (텍스트가 있고 포커스가 없는 경우 = 입력 완료 상태)
    final bool isInputComplete = _inputLogic.hasText && !_inputLogic.isFocused;

    return GestureDetector(
      onTap: () {
        if (widget.showDropdown &&
            widget.items != null &&
            widget.items!.isNotEmpty) {
          _showBottomSheet();
        } else if (widget.onTap != null) {
          widget.onTap!();
        }
      },
      child: Container(
        width: widget.width ?? screenWidth * 0.9,
        height: widget.height ?? 50,
        padding: const EdgeInsets.symmetric(horizontal: 15),
        decoration: BoxDecoration(
          border: Border.all(color: borderColor, width: 1.5),
          borderRadius: BorderRadius.circular(30),
          color: AppColors.whiteLight,
        ),
        child: Row(
          children: [
            // 라벨과 입력 필드 영역
            Expanded(
              child: isInputComplete
                  ? _buildCompletedInput()
                  : _buildEditingInput(),
            ),

            // 드롭다운 아이콘 (showDropdown이 true일 때만 표시)
            if (widget.showDropdown)
              Icon(
                Icons.expand_circle_down_outlined,
                size: 15,
                color: AppColors.textPrimary,
              ),
          ],
        ),
      ),
    );
  }

  // 에러 텍스트 영역
  Widget _buildErrorText(BuildContext context) {
    final double screenWidth = MediaQuery.of(context).size.width;

    if (widget.errorText != null) {
      return Container(
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
      );
    } else {
      return const SizedBox(height: 23);
    }
  }

  // 입력 완료 상태의 UI (라벨 + 값)
  Widget _buildCompletedInput() {
    return Row(
      children: [
        // 라벨이 있는 경우 라벨 표시
        if (widget.label != null)
          SizedBox(
            width: widget.labelWidth ?? 80,
            child: Text(
              widget.label!,
              style: AppTextStyles.bodySmall.copyWith(
                color: AppColors.textLight,
              ),
            ),
          ),

        // 라벨과 입력값 사이의 간격
        if (widget.label != null) const SizedBox(width: 20),

        // 입력된 값 표시 영역
        Expanded(
          child: widget.showDropdown
              ? _buildReadOnlyField() // 드롭다운 표시 시 읽기 전용 필드 사용
              : _buildTextField(),
        ),
      ],
    );
  }

  // 입력 중 또는 초기 상태의 UI (힌트 또는 입력 중)
  Widget _buildEditingInput() {
    return widget.showDropdown
        ? _buildReadOnlyField() // 드롭다운 표시 시 읽기 전용 필드 사용
        : _buildTextField();
  }

  // 텍스트 필드 위젯
  Widget _buildTextField() {
    return TextField(
      controller: _inputLogic.controller,
      focusNode: _inputLogic.focusNode,
      onChanged: (value) {
        if (widget.onChanged != null) {
          widget.onChanged!(value);
        }
      },
      onSubmitted: (value) {
        widget.onSubmitted?.call(value);
      },
      onTap: widget.onTap,
      cursorColor: AppColors.textPrimary,
      decoration: InputDecoration(
        border: InputBorder.none,
        focusedBorder: InputBorder.none,
        enabledBorder: InputBorder.none,
        errorBorder: InputBorder.none,
        disabledBorder: InputBorder.none,
        hintText: widget.hintText,
        hintStyle: AppTextStyles.bodySmall.copyWith(
          color: AppColors.textPrimary.withOpacity(0.5),
        ),
        contentPadding: EdgeInsets.zero,
      ),
      style: AppTextStyles.bodyMediumLight,
      textAlign: TextAlign.left,
    );
  }

  // 읽기 전용 필드 (showDropdown=true일 때 사용)
  Widget _buildReadOnlyField() {
    // 로직 클래스를 사용하여 텍스트와 스타일 결정
    final String displayText = _inputLogic.getDisplayText(widget.hintText);

    final TextStyle textStyle = _inputLogic.getTextStyle(
      AppTextStyles.bodyMediumLight,
      AppTextStyles.bodySmall.copyWith(
        color: AppColors.textPrimary.withOpacity(0.5),
      ),
    );

    return Container(
      alignment: Alignment.centerLeft,
      child: Text(
        displayText,
        style: textStyle,
        overflow: TextOverflow.ellipsis,
      ),
    );
  }
}
