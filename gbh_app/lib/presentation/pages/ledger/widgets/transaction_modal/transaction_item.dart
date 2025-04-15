import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_slidable/flutter_slidable.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/data/models/ledger/category/transaction_category.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_detail_modal.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';

class TransactionListItem extends ConsumerStatefulWidget {
  final Transaction transaction;
  final Function(Transaction)? onDelete;
  final SlidableController? controller;

  const TransactionListItem(
      {Key? key, required this.transaction, this.onDelete, this.controller})
      : super(key: key);

  @override
  ConsumerState<TransactionListItem> createState() =>
      _TransactionListItemState();
}

class _TransactionListItemState extends ConsumerState<TransactionListItem> {
  bool _isDeleting = false;

  @override
  Widget build(BuildContext context) {
    // 삭제 중이면 빈 컨테이너 반환
    if (_isDeleting) {
      return const SizedBox.shrink();
    }

    final categoryRepository = ref.watch(ledgerRepositoryProvider);
    final numberFormat = NumberFormat('#,###', 'ko_KR');

    String iconPath = '';
    if (widget.transaction.type == TransactionType.withdrawal) {
      final category = categoryRepository
          .getWithdrawalCategoryByName(widget.transaction.categoryId);
      iconPath = category?.iconPath ?? '';
    } else if (widget.transaction.type == TransactionType.deposit) {
      final category = categoryRepository
          .getDepositCategoryByName(widget.transaction.categoryId);
      iconPath = category?.iconPath ?? '';
    } else if (widget.transaction.type == TransactionType.transfer) {
      final category = categoryRepository
          .getTransferCategoryByName(widget.transaction.categoryId);
      iconPath = category?.iconPath ?? '';
    }

    // Slidable 위젯으로 감싸기
    return Slidable(
      key: ValueKey(widget.transaction.id),
      // 컨트롤러 연결
      controller: widget.controller,

      // 왼쪽에서 오른쪽으로만 스와이프 가능하도록 설정
      endActionPane: ActionPane(
        motion: const ScrollMotion(),
        extentRatio: 0.2,
        dismissible: DismissiblePane(
          key: ValueKey('dismiss-${widget.transaction.id}'),
          onDismissed: () {
            // 상태 변경을 먼저 해서 위젯을 UI에서 제거
            setState(() {
              _isDeleting = true;
            });

            // 약간의 지연 후 삭제 콜백 실행
            Future.microtask(() {
              if (widget.onDelete != null) {
                widget.onDelete!(widget.transaction);
              }
            });
          },
          closeOnCancel: true,
          confirmDismiss: () async {
            return true;
          },
        ),
        children: [
          CustomSlidableAction(
            onPressed: (context) {
              // 상태 변경하여 위젯을 UI에서 제거
              setState(() {
                _isDeleting = true;
              });

              // 삭제 콜백 실행
              Future.microtask(() {
                if (widget.onDelete != null) {
                  widget.onDelete!(widget.transaction);
                }
              });
            },
            backgroundColor: AppColors.warnning,
            flex: 1,
            child: Icon(
              Icons.delete,
              color: AppColors.background,
            ),
          ),
        ],
      ),

      // 원래 컨텐츠 (변경 없음)
      child: GestureDetector(
        onTap: () {
          showCustomModal(
            context: context,
            ref: ref,
            backgroundColor: AppColors.background,
            child: TransactionDetailModal(
              householdPk: widget.transaction.householdPk,
            ),
          );
        },
        child: Container(
          margin: const EdgeInsets.only(bottom: 8),
          child: Row(
            children: [
              // 카테고리 아이콘
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: AppColors.whiteLight,
                  borderRadius: BorderRadius.circular(15),
                  border:
                      Border.all(color: AppColors.textLight.withOpacity(0.2)),
                ),
                child: iconPath.isNotEmpty
                    ? Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: SvgPicture.asset(
                          iconPath,
                          width: 40,
                          height: 40,
                        ),
                      )
                    : const Icon(Icons.help_outline, size: 24),
              ),
              const SizedBox(width: 12),

              // 거래 내용
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.transaction.title,
                      style: AppTextStyles.bodyMedium.copyWith(
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      '${widget.transaction.householdCategory} | ${widget.transaction.paymentMethod}',
                      style: AppTextStyles.bodySmall.copyWith(
                        color: AppColors.textSecondary,
                        fontWeight: FontWeight.w300,
                      ),
                    ),
                  ],
                ),
              ),

              // 금액
              Text(
                widget.transaction.type == TransactionType.withdrawal
                    ? '- ${numberFormat.format(widget.transaction.amount)}원'
                    : widget.transaction.type == TransactionType.deposit
                        ? '+ ${numberFormat.format(widget.transaction.amount)}원'
                        : '${numberFormat.format(widget.transaction.amount)}원',
                style: widget.transaction.type == TransactionType.withdrawal
                    ? AppTextStyles.bodyLarge.copyWith(
                        fontSize: 16,
                        color: AppColors.textPrimary,
                      )
                    : widget.transaction.type == TransactionType.deposit
                        ? AppTextStyles.bodyLarge.copyWith(
                            fontSize: 16,
                            color: AppColors.blueDark,
                          )
                        : AppTextStyles.bodyLarge.copyWith(
                            fontSize: 16,
                            color: AppColors.textSecondary,
                          ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
