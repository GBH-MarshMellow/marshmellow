import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_field.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/editable_memo_filed.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/date_time_wheel_picker.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_category_picker.dart';

class PortfolioFields {
  // 날짜 필드
  static PortfolioField dateField({
    required DateTime selectedDate,
    required BuildContext context,
    required WidgetRef ref,
    required Function(DateTime) onDateChanged,
    bool includeTime = true,
  }) {
    // 날짜 포맷
    String formattedDate =
        '${selectedDate.year}년 ${selectedDate.month.toString().padLeft(2, '0')}월 ${selectedDate.day.toString().padLeft(2, '0')}일';

    // 시간 포함 시 포맷 변경
    if (includeTime) {
      formattedDate +=
          ' ${selectedDate.hour.toString().padLeft(2, '0')}:${selectedDate.minute.toString().padLeft(2, '0')}';
    }

    return PortfolioField(
      label: includeTime ? '날짜 및 시간' : '날짜',
      trailing: Text(formattedDate, style: AppTextStyles.bodySmall),
      onTap: () {
        showDateTimePickerBottomSheet(
          context: context,
          ref: ref,
          initialDateTime: selectedDate,
          onDateTimeChanged: onDateChanged,
          initialMode: CupertinoDatePickerMode.date,
          confirmButtonText: includeTime ? '확인' : '선택',
          nextButtonText: '다음',
        );
      },
    );
  }

  // 카테고리 필드
  static PortfolioField categoryField({
    required BuildContext context,
    required WidgetRef ref, // ref 필요
    String? selectedCategory,
    required Function(String, int) onCategorySelected, // 두 매개변수 필요
  }) {
    return PortfolioField(
      label: '카테고리',
      value: selectedCategory,
      onTap: () {
        showModalBottomSheet(
          context: context,
          isScrollControlled: true,
          backgroundColor: Colors.transparent,
          builder: (context) => PortfolioCategoryPicker(
            selectedCategory: selectedCategory ?? "",
            onCategorySelected: onCategorySelected,
          ),
        );
      },
      valueStyle: AppTextStyles.bodySmall,
    );
  }

  // 파일명 필드
  static Widget editableFileNameField({
    String? fileName,
    required Function(String) onFileNameChanged,
  }) {
    return EditableMemoField(
      label: '파일명',
      initialValue: fileName,
      onChanged: onFileNameChanged,
    );
  }

  // 메모/키워드 필드
  static Widget editableMemoField({
    String? memo,
    required Function(String) onMemoChanged,
  }) {
    return EditableMemoField(
      label: '메모/키워드',
      initialValue: memo,
      onChanged: onMemoChanged,
    );
  }
}
