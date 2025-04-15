import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/transaction_field.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/transaction_fields.dart';
import 'package:marshmellow/data/models/ledger/category/transfer_category.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/transfer_direction_picker.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/category/category_mapping.dart';
import 'package:marshmellow/data/models/ledger/payment_method.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/payment_method_picker.dart';

class TransferForm extends ConsumerStatefulWidget {
  final Transaction? initialData; // 초기 데이터 추가
  final DateTime? initialDate; // 초기 날짜
  final Function(int? amount, String? memo, int? categoryPk)?
      onDataChanged; // 콜백 함수 수정
  final bool readOnly; // 읽기 전용 모드

  const TransferForm({
    super.key,
    this.initialData,
    this.initialDate,
    this.onDataChanged,
    this.readOnly = false,
  });

  @override
  ConsumerState<TransferForm> createState() => TransferFormState();
}

class TransferFormState extends ConsumerState<TransferForm> {
  DateTime _selectedDate = DateTime.now();
  String? _merchant;
  String? _memo;
  TransferCategory? _selectedTransferCategory;
  TransferDirection? _transferDirection;
  String? _account;
  int? _amount;

  @override
  void initState() {
    super.initState();
    // 초기 데이터가 있으면 설정
    if (widget.initialData != null) {
      final transaction = widget.initialData!;
      final categoryRepository = ref.read(ledgerRepositoryProvider);

      // 트랜잭션 데이터로 폼 초기화
      _selectedDate = transaction.dateTime;
      _merchant = transaction.tradeName;
      _memo = transaction.householdMemo;
      _account = transaction.paymentMethod;
      _amount = transaction.householdAmount;

      // 카테고리 설정
      _selectedTransferCategory = categoryRepository
          .getTransferCategoryByName(transaction.householdCategory);

      // 트랜잭션 방향 설정 - 기본적으로 출금으로 설정하고, 필요하면 수정
      _transferDirection = TransferDirection.withdrawal;
    } else {
      // initialDate가 있으면 그 날짜를 사용, 없으면 현재 날짜 사용
      _selectedDate = widget.initialDate ?? DateTime.now();
    }
  }

  // 날짜 업데이트 함수
  void _updateDate(DateTime date) {
    setState(() {
      _selectedDate = date;
    });
  }

  // 메모 업데이트 함수
  void _updateMemo(String value) {
    setState(() {
      _memo = value;
    });
    _notifyDataChanged();
  }

  // 상호명 업데이트 함수
  void _updateMerchant(String value) {
    setState(() {
      _merchant = value;
    });
  }

  // 이체 방향 및 카테고리 업데이트 함수
  void _updateTransferInfo(
      TransferDirection direction, TransferCategory category) {
    setState(() {
      _transferDirection = direction;
      _selectedTransferCategory = category;
    });

    // 카테고리 PK 가져오기
    final pk = CategoryPkMapping.getPkFromCategory(
        transferCategory: category, transferDirection: direction);
    _notifyDataChanged(categoryPk: pk);
  }

  // 계좌 업데이트 함수 추가
  void _updateAccount(PaymentMethod method) {
    setState(() {
      _account = method.paymentMethod;
    });
  }

  // 카테고리 선택 모달 표시
  void _showCategorySelectionModal() async {
    final result = await showTransferDirectionPickerModal(context);
    if (result != null) {
      _updateTransferInfo(result['direction'], result['category']);
    }
  }

  // 데이터 변경을 알리는 함수 수정
  void _notifyDataChanged({int? categoryPk}) {
    if (widget.onDataChanged != null) {
      int? pkToUse = categoryPk;
      // categoryPk가 없으면 현재 선택된 카테고리와 방향의 PK 사용
      if (pkToUse == null &&
          _selectedTransferCategory != null &&
          _transferDirection != null) {
        pkToUse = CategoryPkMapping.getPkFromCategory(
            transferCategory: _selectedTransferCategory,
            transferDirection: _transferDirection);
      }

      widget.onDataChanged!(_amount, _memo, pkToUse);
    }
  }

  Map<String, dynamic> getFormData() {
    return {
      'date': _selectedDate,
      'tradeName': _merchant,
      'paymentMethod': _account,
      'memo': _memo,
      'categoryPk':
          (_selectedTransferCategory != null && _transferDirection != null)
              ? CategoryPkMapping.getPkFromCategory(
                  transferCategory: _selectedTransferCategory,
                  transferDirection: _transferDirection)
              : null,
    };
  }

  @override
  Widget build(BuildContext context) {
    // 카테고리 표시 텍스트
    String categoryDisplayText = '';
    if (_transferDirection != null && _selectedTransferCategory != null) {
      String directionText =
          _transferDirection == TransferDirection.deposit ? '입금' : '출금';
      categoryDisplayText =
          '$directionText > ${_selectedTransferCategory!.name}';
    }

    return Column(
      children: [
        // 카테고리 필드
        TransactionField(
          label: '카테고리',
          value: categoryDisplayText,
          onTap: _showCategorySelectionModal,
          valueStyle: AppTextStyles.bodySmall.copyWith(
            color: AppColors.textPrimary,
          ),
        ),

        // 상호명 필드
        TransactionFields.editableMerchantField(
          merchantName: _merchant,
          onMerchantChanged: _updateMerchant,
          enabled: !widget.readOnly,
        ),

        // 계좌 필드
        TransactionField(
          label: '계좌',
          value: _account,
          onTap: widget.readOnly
              ? null
              : () {
                  showPaymentMethodPickerModal(
                    context,
                    transactionType: 'transfer',
                    onPaymentMethodSelected: _updateAccount,
                    title: '계좌 선택',
                  );
                },
          valueStyle: widget.readOnly
              ? AppTextStyles.bodySmall.copyWith(color: AppColors.textSecondary)
              : null,
        ),
        // 날짜 필드
        TransactionFields.dateField(
          context: context,
          ref: ref,
          selectedDate: _selectedDate,
          onDateChanged: _updateDate,
          enabled: !widget.readOnly,
        ),

        // 메모 필드
        TransactionFields.editableMemoField(
          memo: _memo,
          onMemoChanged: _updateMemo,
          enabled: true, // 메모는 항상 수정 가능
        ),
      ],
    );
  }
}
