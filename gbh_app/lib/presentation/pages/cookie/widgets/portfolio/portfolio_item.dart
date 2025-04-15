import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_slidable/flutter_slidable.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_model.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_edit_modal.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';

class PortfolioItem extends StatelessWidget {
  final PortfolioModel portfolio;
  final VoidCallback? onTap;
  final Function(PortfolioModel)? onDelete;
  final SlidableController? controller;
  final bool isSelectionMode;
  final bool isSelected;
  final VoidCallback? onSelectionToggle;
  final VoidCallback? onLongPress;

  const PortfolioItem({
    Key? key,
    required this.portfolio,
    this.onTap,
    this.onDelete,
    this.controller,
    this.isSelectionMode = false,
    this.isSelected = false,
    this.onSelectionToggle,
    this.onLongPress,
  }) : super(key: key);

  // 파일 확장자에 따른 아이콘을 선택하는 메서드
  String _selectFileIcon(String fileName) {
    final extension = _extractFileExtension(fileName);

    switch (extension) {
      case 'pdf':
        return IconPath.filePdf;
      case 'doc':
      case 'docx':
        return IconPath.fileDoc;
      case 'xls':
      case 'xlsx':
        return IconPath.fileXls;
      case 'ppt':
      case 'pptx':
        return IconPath.filePpt;
      case 'jpg':
      case 'jpeg':
        return IconPath.fileJpg;
      case 'png':
        return IconPath.filePng;
      case 'svg':
        return IconPath.fileSvg;
      case 'txt':
        return IconPath.fileTxt;
      case 'zip':
      case 'rar':
        return IconPath.fileZip;
      case 'csv':
        return IconPath.fileCsv;
      default:
        return IconPath.fileText;
    }
  }

  // 안전하게 파일 확장자를 추출하는 메서드
  String _extractFileExtension(String fileName) {
    try {
      final parts = fileName.split('.');
      return parts.isNotEmpty ? parts.last.toLowerCase() : '';
    } catch (e) {
      return '';
    }
  }

  @override
  Widget build(BuildContext context) {
    final itemContent = GestureDetector(
      onTap: () {
        // 선택 모드일 때는 onSelectionToggle 호출
        if (isSelectionMode && onSelectionToggle != null) {
          onSelectionToggle!();
        }
        // 선택 모드가 아니고 onTap이 있을 때 호출
        else if (!isSelectionMode && onTap != null) {
          onTap!();
        }
      },
      onLongPress: onLongPress,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
        child: Row(
          children: [
            SvgPicture.asset(
              _selectFileIcon(portfolio.originFileName),
              width: 30,
              height: 30,
              fit: BoxFit.contain,
              colorFilter: const ColorFilter.mode(
                AppColors.textPrimary,
                BlendMode.srcIn,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    portfolio.fileName,
                    style: AppTextStyles.bodyMedium,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    portfolio.portfolioMemo,
                    style: AppTextStyles.bodySmall.copyWith(
                      color: AppColors.textSecondary,
                      fontWeight: FontWeight.w300,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
            // 선택 모드일 때 체크 아이콘 표시
            if (isSelectionMode)
              SvgPicture.asset(
                isSelected ? IconPath.checked : IconPath.unchecked,
                width: 20,
                height: 20,
              ),
          ],
        ),
      ),
    );

    if (!isSelectionMode) {
      return Slidable(
        key: ValueKey(portfolio.portfolioPk),
        controller: controller,
        endActionPane: ActionPane(
          motion: const ScrollMotion(),
          extentRatio: 0.2,
          dismissible: DismissiblePane(
            key: ValueKey('dismiss-${portfolio.portfolioPk}'),
            onDismissed: () {
              if (onDelete != null) {
                onDelete!(portfolio);
              }
            },
            closeOnCancel: true,
            confirmDismiss: () async {
              return true;
            },
          ),
          children: [
            CustomSlidableAction(
              onPressed: (context) {
                if (onDelete != null) {
                  onDelete!(portfolio);
                }
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
        child: itemContent,
      );
    }

    return itemContent;
  }
}
