import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/main/ledger_transaction_history.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/main/ledger_calendar.dart';

class PageDotIndicator extends StatelessWidget {
  final int currentPage;
  final int totalPages;
  final PageController pageController;
  final Function(int) onPageChanged;

  const PageDotIndicator({
    super.key,
    required this.currentPage,
    required this.totalPages,
    required this.pageController,
    required this.onPageChanged,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      // 수평 스와이프 감지
      onHorizontalDragEnd: (details) {
        // 스와이프 방향에 따라 페이지 이동
        if (details.primaryVelocity! > 0) {
          // 오른쪽으로 스와이프 (이전 페이지)
          if (currentPage > 0) {
            final newPage = currentPage - 1;
            pageController.animateToPage(
              newPage,
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeInOut,
            );
            onPageChanged(newPage);
          }
        } else {
          // 왼쪽으로 스와이프 (다음 페이지)
          if (currentPage < totalPages - 1) {
            final newPage = currentPage + 1;
            pageController.animateToPage(
              newPage,
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeInOut,
            );
            onPageChanged(newPage);
          }
        }
      },
      // 페이지 인디케이터 점들
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: List.generate(totalPages, (index) {
          return Container(
            margin: const EdgeInsets.symmetric(horizontal: 2),
            width: 5,
            height: 5,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: index == currentPage
                  ? AppColors.textPrimary
                  : AppColors.whitePrimary,
              border: Border.all(
                color: AppColors.textPrimary,
                width: 1,
              ),
            ),
          );
        }),
      ),
    );
  }
}
