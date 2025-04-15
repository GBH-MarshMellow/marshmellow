import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

// 상태관리
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/di/providers/modal_provider.dart';

/// 모달 위젯
class Modal extends ConsumerWidget {
  final Color backgroundColor;
  final Widget child;
  final EdgeInsetsGeometry padding;
  final double? maxHeight;
  final String? title;
  final TextStyle? titleStyle;
  final bool showDivider;

  const Modal({
    Key? key,
    required this.backgroundColor,
    required this.child,
    this.padding = const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
    this.maxHeight,
    this.title,
    this.titleStyle,
    this.showDivider = true,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final screenHeight = MediaQuery.of(context).size.height;
    final screenWidth = MediaQuery.of(context).size.width;
    final effectiveMaxHeight = maxHeight ?? screenHeight * 0.8;

    return Container(
      constraints: BoxConstraints(
        maxHeight: effectiveMaxHeight,
      ),
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(15)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 핸들
          Center(
            child: Padding(
              padding: const EdgeInsets.only(top: 12),
              child: Container(
                width: screenWidth * 0.1,
                height: screenWidth * 0.01,
                decoration: BoxDecoration(
                  color: AppColors.textPrimary,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
          ),

          // 제목이 있는 경우에만 표시
          if (title != null) ...[
            Padding(
              padding: const EdgeInsets.only(
                top: 12,
                left: 25,
              ),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  title!,
                  style: titleStyle ?? AppTextStyles.bodyMedium,
                  textAlign: TextAlign.left,
                ),
              ),
            ),

            // 구분선 표시 옵션이 활성화된 경우에만 표시
            if (showDivider)
              Divider(
                height: 1,
                thickness: 0.5,
                color: AppColors.textSecondary.withOpacity(0.3),
              ),

            const SizedBox(height: 8), // 제목과 내용 사이 간격
          ],

          // 기존 내용
          Flexible(
            child: Padding(
              padding: EdgeInsets.only(
                top: title != null ? 0 : padding.vertical / 2,
                bottom: padding.vertical / 2,
                left: padding.horizontal / 2,
                right: padding.horizontal / 2,
              ),
              child: child,
            ),
          ),
        ],
      ),
    );
  }
}

// EdgeInsetsGeometry에 horizontal과 vertical 속성 추가 (편의성)
extension EdgeInsetsGeometryExtension on EdgeInsetsGeometry {
  double get horizontal {
    if (this is EdgeInsets) {
      final edgeInsets = this as EdgeInsets;
      return edgeInsets.left + edgeInsets.right;
    }
    return 32.0; // 기본값
  }

  double get vertical {
    if (this is EdgeInsets) {
      final edgeInsets = this as EdgeInsets;
      return edgeInsets.top + edgeInsets.bottom;
    }
    return 40.0; // 기본값
  }
}

/// 범용 모달을 표시하는 함수
void showCustomModal({
  required BuildContext context,
  required Widget child,
  WidgetRef? ref,
  String? title,
  TextStyle? titleStyle,
  Color backgroundColor = Colors.white,
  EdgeInsetsGeometry? padding,
  double? maxHeight,
  bool showDivider = true,
}) {
  // 모달 상태 변경
  ref?.read(modalProvider.notifier).showModal();

  showModalBottomSheet(
    context: context,
    backgroundColor: Colors.transparent,
    isScrollControlled: true,
    builder: (BuildContext context) {
      return Modal(
        backgroundColor: backgroundColor,
        padding:
            padding ?? const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
        maxHeight: maxHeight,
        title: title,
        titleStyle: titleStyle,
        showDivider: showDivider,
        child: child,
      );
    },
  ).then((_) {
    // 모달이 닫힐 때 상태 업데이트
    ref?.read(modalProvider.notifier).hideModal();
  });
}
