import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/category/withdrawal_category.dart';
import 'package:marshmellow/data/models/ledger/category/deposit_category.dart';
import 'package:marshmellow/data/models/ledger/category/transfer_category.dart';

import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/transaction_field.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/editable_memo_filed.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/date_time_wheel_picker.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/expense_category_picker.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/income_category_picker.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/transfer_category_picker.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/transfer_direction_picker.dart';

// 지출 카테고리 선택 모달 함수
Future<void> showExpenseCategoryPickerModal(
  BuildContext context, {
  required Function(WithdrawalCategory) onCategorySelected,
}) {
  return showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (context) => ExpenseCategoryPicker(
      onCategorySelected: onCategorySelected,
    ),
  );
}

// 수입 카테고리 선택 모달 함수
Future<void> showIncomeCategoryPickerModal(
  BuildContext context, {
  required Function(DepositCategory) onCategorySelected,
}) {
  return showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (context) => IncomeCategoryPicker(
      onCategorySelected: onCategorySelected,
    ),
  );
}

// TransactionFields 클래스
class TransactionFields {
  // 날짜 필드
  static TransactionField dateField({
    required DateTime selectedDate,
    required BuildContext context,
    required WidgetRef ref,
    required Function(DateTime) onDateChanged,
    bool includeTime = true, // 시간 포함 여부 옵션
    bool enabled = true,
  }) {
    // 날짜 포맷
    String formattedDate =
        '${selectedDate.year}년 ${selectedDate.month.toString().padLeft(2, '0')}월 ${selectedDate.day.toString().padLeft(2, '0')}일';

    // 시간 포함 시 포맷 변경
    if (includeTime) {
      formattedDate +=
          ' ${selectedDate.hour.toString().padLeft(2, '0')}:${selectedDate.minute.toString().padLeft(2, '0')}';
    }

    return TransactionField(
      label: includeTime ? '날짜 및 시간' : '날짜',
      trailing: Text(
        formattedDate,
        style: AppTextStyles.bodySmall.copyWith(
          color: enabled ? AppColors.textPrimary : AppColors.textSecondary,
        ),
      ),
      onTap: enabled
          ? () {
              showDateTimePickerBottomSheet(
                context: context,
                ref: ref,
                initialDateTime: selectedDate,
                onDateTimeChanged: onDateChanged,
                initialMode: CupertinoDatePickerMode.date,
                confirmButtonText: includeTime ? '확인' : '선택',
                nextButtonText: '다음',
              );
            }
          : null,
    );
  }

  // 지출 카테고리 필드
  static TransactionField expenseCategoryField({
    required BuildContext context,
    String? selectedCategory,
    Function(WithdrawalCategory)? onCategorySelected,
    bool enabled = true, // 활성화 여부 파라미터 추가
  }) {
    return TransactionField(
      label: '카테고리',
      value: selectedCategory,
      onTap: enabled
          ? () {
              if (onCategorySelected != null) {
                showExpenseCategoryPickerModal(
                  context,
                  onCategorySelected: onCategorySelected,
                );
              }
            }
          : null, // enabled가 false면 onTap이 null이므로 터치 불가
      // 항상 일관된 스타일 적용
      valueStyle: AppTextStyles.bodySmall.copyWith(
        color: enabled ? AppColors.textPrimary : AppColors.textSecondary,
      ),
    );
  }

  // 수입 카테고리 필드
  static TransactionField incomeCategoryField({
    required BuildContext context,
    String? selectedCategory,
    required Function(DepositCategory) onCategorySelected,
    bool enabled = true,
  }) {
    return TransactionField(
      label: '카테고리',
      value: selectedCategory,
      onTap: enabled
          ? () {
              showIncomeCategoryPickerModal(
                context,
                onCategorySelected: onCategorySelected,
              );
            }
          : null,
      valueStyle: AppTextStyles.bodySmall.copyWith(
        color: enabled ? AppColors.textPrimary : AppColors.textSecondary,
      ),
    );
  }

  // 이체 카테고리 필드
  static TransactionField transferCategoryField({
    required BuildContext context,
    String? selectedCategory,
    required Function(TransferCategory) onCategorySelected,
    required TransferDirection direction,
    bool enabled = true,
  }) {
    return TransactionField(
      label: '카테고리',
      value: selectedCategory,
      onTap: enabled
          ? () async {
              final selectedCategory = await showTransferCategoryPickerModal(
                context,
                direction: direction,
              );

              // 선택된 카테고리가 있으면 콜백 호출
              if (selectedCategory != null) {
                onCategorySelected(selectedCategory);
              }
            }
          : null,
      valueStyle: AppTextStyles.bodySmall.copyWith(
        color: enabled ? AppColors.textPrimary : AppColors.textSecondary,
      ),
    );
  }

  // 상호명 필드 (일반)
  static TransactionField merchantField({
    String? merchantName,
    VoidCallback? onTap,
    bool enabled = true,
  }) {
    return TransactionField(
      label: '상호명',
      value: merchantName,
      onTap: enabled ? onTap : null,
      valueStyle: AppTextStyles.bodySmall.copyWith(
        color: enabled ? AppColors.textPrimary : AppColors.textSecondary,
      ),
    );
  }

  // 상호명 필드 (편집 가능)
  static Widget editableMerchantField({
    String? merchantName,
    required Function(String) onMerchantChanged,
    bool enabled = true,
  }) {
    return EditableMemoField(
      label: '상호명',
      initialValue: merchantName,
      onChanged: onMerchantChanged,
      enabled: enabled,
    );
  }

  // 메모/키워드 필드 (편집 가능)
  static Widget editableMemoField({
    String? memo,
    required Function(String) onMemoChanged,
    bool enabled = true,
  }) {
    return EditableMemoField(
      label: '메모/키워드',
      initialValue: memo,
      onChanged: onMemoChanged,
      enabled: enabled,
    );
  }

  // 결제수단 필드
  static TransactionField paymentMethodField({
    String? method,
    VoidCallback? onTap,
    bool enabled = true,
  }) {
    return TransactionField(
      label: '결제수단',
      value: method,
      onTap: enabled ? onTap : null,
      valueStyle: AppTextStyles.bodySmall.copyWith(
        color: enabled ? AppColors.textPrimary : AppColors.textSecondary,
      ),
    );
  }

  // 입금계좌 필드
  static TransactionField depositAccountField({
    String? account,
    VoidCallback? onTap,
    bool enabled = true,
  }) {
    return TransactionField(
      label: '입금계좌',
      value: account,
      onTap: enabled ? onTap : null,
      valueStyle: AppTextStyles.bodySmall.copyWith(
        color: enabled ? AppColors.textPrimary : AppColors.textSecondary,
      ),
    );
  }

  // 출금계좌 필드
  static TransactionField withdrawalAccountField({
    String? account,
    VoidCallback? onTap,
    bool enabled = true,
  }) {
    return TransactionField(
      label: '계좌',
      value: account,
      onTap: enabled ? onTap : null,
      valueStyle: AppTextStyles.bodySmall.copyWith(
        color: enabled ? AppColors.textPrimary : AppColors.textSecondary,
      ),
    );
  }

  // 예산에서 제외 필드
  static TransactionField excludeFromBudgetField({
    required bool value,
    required Function(bool) onChanged,
    bool enabled = true,
  }) {
    return TransactionField(
      label: '예산에서 제외',
      padding: EdgeInsets.only(top: 10, bottom: 10, left: 10),
      trailing: Transform.scale(
        scale: 0.8,
        child: CupertinoSwitch(
          value: value,
          onChanged: enabled ? onChanged : null,
          thumbColor: AppColors.whiteLight,
          activeColor:
              enabled ? AppColors.textPrimary : AppColors.textSecondary,
        ),
      ),
    );
  }

  // 분류 선택기 필드
  static TransactionField typeSelectorField({
    required Widget selector,
    bool enabled = true,
  }) {
    return TransactionField(
      label: '분류',
      trailing: SizedBox(
        width: 200,
        child: selector,
      ),
      onTap: enabled ? () {} : null,
    );
  }
}
