import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/viewmodels/modal/modal_layout.dart';
import 'package:marshmellow/presentation/widgets/select_input/select_input_logic.dart';

class SelectInput<T> extends StatefulWidget {
  final String label;
  final bool readOnly;
  final ValueChanged<T>? onItemSelected;
  final TextEditingController controller;
  final double? width;
  final List<T> items;
  final ItemBuilder<T> itemBuilder;
  final String Function(T)? displayStringForItem;
  final bool showDividers;
  final String? modalTitle;
  final bool showTitleDivider;

  const SelectInput({
    super.key,
    required this.label,
    required this.controller,
    required this.items,
    required this.itemBuilder,
    this.onItemSelected,
    this.readOnly = true,
    this.width,
    this.displayStringForItem,
    this.showDividers = true,
    this.modalTitle,
    this.showTitleDivider = false,
  });

  @override
  State<SelectInput<T>> createState() => _SelectInputState<T>();
}

class _SelectInputState<T> extends State<SelectInput<T>> {
  late final SelectInputController<T> _controller;

  @override
  void initState() {
    super.initState();
    _controller = SelectInputController<T>(
      textController: widget.controller,
      onItemSelected: widget.onItemSelected,
      displayStringForItem: widget.displayStringForItem,
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _showBottomSheet() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _controller.clearFocus();
    });

    ModalLayout.showSelectionModal<T>(
      context: context,
      items: widget.items,
      displayStringForItem:
          widget.displayStringForItem ?? (item) => item.toString(),
      onItemSelected: (item) {
        _controller.selectItem(item);
        if (widget.onItemSelected != null) {
          widget.onItemSelected!(item);
        }
      },
      modalTitle: widget.modalTitle,
      showDividers: widget.showDividers,
      showTitleDivider: widget.showTitleDivider,
      selectedItem: _controller.selectedItem,
    );
  }

  @override
  Widget build(BuildContext context) {
    final double screenWidth = MediaQuery.of(context).size.width;

    return Center(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          GestureDetector(
            onTap: () {
              FocusScope.of(context).requestFocus(_controller.focusNode);
              _showBottomSheet();
            },
            child: Container(
              width: widget.width ?? screenWidth * 0.9,
              height: 60,
              padding: const EdgeInsets.symmetric(horizontal: 15),
              decoration: BoxDecoration(
                border: Border.all(
                    color: _controller.isFocused
                        ? AppColors.textPrimary
                        : AppColors.textSecondary),
                borderRadius: BorderRadius.circular(5),
                color: AppColors.whiteLight,
              ),
              child: Stack(
                children: [
                  TextField(
                    focusNode: _controller.focusNode,
                    controller: widget.controller,
                    readOnly: true,
                    onTap: _showBottomSheet,
                    cursorColor: _controller.isFocused
                        ? AppColors.textPrimary
                        : AppColors.textSecondary,
                    decoration: InputDecoration(
                      labelText: widget.label,
                      labelStyle: AppTextStyles.bodySmall.copyWith(
                        color: _controller.isFocused
                            ? AppColors.textPrimary
                            : AppColors.textSecondary,
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
                      color: _controller.isFocused
                          ? AppColors.textPrimary
                          : AppColors.textSecondary,
                    ),
                  ),
                  Positioned(
                    right: 0,
                    bottom: 10,
                    child: Icon(Icons.expand_circle_down_outlined,
                        size: 15, color: AppColors.textSecondary),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ===================== 사용 예시 =====================
/*
// SelectInput 위젯 사용 예시

!!!!! 이것만 봐도 되는 간단한 사용법!!!! 

SelectInput<String>(
  label: "국가 선택",
  controller: _countryController,
  items: ["미국", "캐나다", "영국", "호주", "일본", "한국"],
  itemBuilder: (context, item, isSelected) => ListTile(
    title: Text(item),
  ),
  onItemSelected: (value) {
    print("선택된 국가: $value");
  },
), SelectInput<String>(
   label: "국가 선택",
   modalTitle: "국가 선택",
   controller: _countryController,
   items: ["미국", "캐나다", "영국", "호주", "일본", "한국"],
   itemBuilder: (context, item, isSelected) =>
      ListTile(title: Text(item)),
   onItemSelected: (value) {
      print("선택된 국가: $value");
      },
       ),

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

*/
