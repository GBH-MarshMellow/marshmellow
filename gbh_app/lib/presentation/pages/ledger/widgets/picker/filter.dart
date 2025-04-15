import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/di/providers/transaction_filter_provider.dart';

class TransactionFilterDropdown extends ConsumerStatefulWidget {
  final Function(TransactionFilterType) onFilterSelected;

  const TransactionFilterDropdown({
    Key? key,
    required this.onFilterSelected,
  }) : super(key: key);

  @override
  _TransactionFilterDropdownState createState() =>
      _TransactionFilterDropdownState();
}

class _TransactionFilterDropdownState
    extends ConsumerState<TransactionFilterDropdown> {
  @override
  Widget build(BuildContext context) {
    // 현재 선택된 필터 가져오기
    final currentFilter = ref.watch(transactionFilterProvider);

    return Material(
      color: Colors.transparent,
      child: Container(
        width: 100,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(8),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.3),
              spreadRadius: 1,
              blurRadius: 5,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: TransactionFilterType.values.map((filter) {
            return InkWell(
              onTap: () {
                // 필터 상태 업데이트
                ref.read(transactionFilterProvider.notifier).state = filter;

                // 외부 콜백 호출
                widget.onFilterSelected(filter);

                // 드롭다운 닫기
                Navigator.of(context).pop();
              },
              child: Padding(
                padding:
                    const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                child: Row(
                  children: [
                    SvgPicture.asset(
                      currentFilter == filter
                          ? IconPath.filtered
                          : IconPath.unfiltered,
                      width: 14,
                      height: 14,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      filter.label,
                      style: AppTextStyles.bodyMedium.copyWith(
                        color: currentFilter == filter
                            ? AppColors.textPrimary
                            : AppColors.textSecondary,
                        fontWeight: currentFilter == filter
                            ? FontWeight.w600
                            : FontWeight.w400,
                      ),
                    ),
                  ],
                ),
              ),
            );
          }).toList(),
        ),
      ),
    );
  }
}

// 드롭다운을 표시하는 확장 함수
extension ShowDropdown on BuildContext {
  void showTransactionFilterDropdown({
    required GlobalKey dropdownKey,
    required Function(TransactionFilterType) onFilterSelected,
  }) {
    final RenderBox renderBox =
        dropdownKey.currentContext!.findRenderObject() as RenderBox;
    final size = renderBox.size;
    final position = renderBox.localToGlobal(Offset.zero);

    final overlayEntry = OverlayEntry(
      builder: (context) => Stack(
        children: [
          Positioned.fill(
            child: GestureDetector(
              onTap: () => Navigator.of(context).pop(),
              child: Container(color: Colors.transparent),
            ),
          ),
          Positioned(
            left: position.dx,
            top: position.dy + size.height + 5, // 버튼 아래에 위치
            child: TransactionFilterDropdown(
              onFilterSelected: onFilterSelected,
            ),
          ),
        ],
      ),
    );

    Navigator.of(this).push(
      PageRouteBuilder(
        opaque: false,
        pageBuilder: (context, _, __) => Scaffold(
          backgroundColor: Colors.transparent,
          body: Stack(
            children: [overlayEntry.builder!(context)],
          ),
        ),
      ),
    );
  }
}
