import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_field.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/editable_memo_filed.dart';

class CategoryFields {
  // 카테고리명 필드 (편집 가능)
  static Widget editableCategoryNameField({
    String? categoryName,
    required Function(String) onCategoryNameChanged,
  }) {
    return EditableMemoField(
      label: '카테고리명',
      initialValue: categoryName,
      onChanged: onCategoryNameChanged,
    );
  }

  // 메모 필드 (편집 가능)
  static Widget editableMemoField({
    String? memo,
    required Function(String) onMemoChanged,
  }) {
    return EditableMemoField(
      label: '메모',
      initialValue: memo,
      onChanged: onMemoChanged,
    );
  }
}
