import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';

class ModalLayout {
  static Future<T?> showSelectionModal<T>({
    required BuildContext context,
    required List<T> items,
    required String Function(T) displayStringForItem,
    required ValueChanged<T> onItemSelected,
    String? modalTitle,
    bool showDividers = true,
    bool showTitleDivider = false,
    T? selectedItem,
  }) {
    return showModalBottomSheet<T>(
      context: context,
      isScrollControlled: true,
      useRootNavigator: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return Modal(
          backgroundColor: AppColors.whiteLight,
          title: modalTitle,
          showDivider: showTitleDivider,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Flexible(
                child: ListView.separated(
                  shrinkWrap: true,
                  itemCount: items.length,
                  separatorBuilder: (context, index) {
                    return showDividers
                        ? Divider(
                            height: 1,
                            thickness: 0.5,
                            color: AppColors.textSecondary.withOpacity(0.3),
                          )
                        : const SizedBox.shrink();
                  },
                  itemBuilder: (context, index) {
                    final item = items[index];
                    final isSelected = selectedItem == item;

                    return InkWell(
                      onTap: () {
                        onItemSelected(item);
                        Navigator.pop(context, item);
                      },
                      child: Container(
                        alignment: Alignment.centerLeft,
                        padding: const EdgeInsets.symmetric(
                          vertical: 12,
                          horizontal: 15,
                        ),
                        child: Text(
                          displayStringForItem(item),
                          style: AppTextStyles.bodyLargeLight.copyWith(
                            fontSize: 15,
                          ),
                          textAlign: TextAlign.left,
                        ),
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
