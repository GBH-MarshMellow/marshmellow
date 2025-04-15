// presentation/pages/finance/widgets/banner_ad_widget.dart
import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:flutter_svg/flutter_svg.dart';

class BannerAdWidget extends StatelessWidget {
  final VoidCallback? onClose; // 닫기 콜백 추가
  final ScrollController? scrollController; // 스크롤 컨트롤러 추가
  
  const BannerAdWidget({
    Key? key, 
    this.onClose,
    this.scrollController, // 스크롤 컨트롤러 파라미터 추가
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
  return Stack(
    children: [
      GestureDetector(
          onTap: () {
            // 스크롤 컨트롤러가 있으면 최하단으로 스크롤
            if (scrollController != null) {
              scrollController!.animateTo(
                scrollController!.position.maxScrollExtent,
                duration: const Duration(milliseconds: 500),
                curve: Curves.easeOut,
              );
            }
          },
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: AppColors.bluePrimary,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Row(
            children: [
              Container(
                decoration: BoxDecoration(
                  color: AppColors.whiteLight,
                  borderRadius: BorderRadius.all(Radius.circular(4.0))
                ),
                width: 60,
                height: 60,
                child: ClipRRect(
                  borderRadius: BorderRadius.all(Radius.circular(4.0)),
                  child: Transform.scale(
                    scale: 1.5, // 2배 확대
                    alignment: Alignment(0, 0), // 살짝 오른쪽 중앙 부분이 보이도록
                    child: Lottie.asset(
                      'assets/images/loading/test_lottie.json',
                      fit: BoxFit.contain, // 여기서는 contain으로 돌려놓음
                    ),
                  ),
                ),
              ),   
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      '내 대출',
                      style: AppTextStyles.bodySmall
                    ),
                    const Text(
                      '한번에 관리해볼까?',
                      style: AppTextStyles.bodyExtraSmall
                    ),
                    const SizedBox(height: 4),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        Text(
                          '대출 확인하기 >',
                          style: AppTextStyles.moneyBodySmall
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
      Positioned(
        top: -8,
        right: -8,
        child: IconButton(
          icon: SvgPicture.asset(IconPath.exitgray,
            width: 16,
            height: 16,
          ),
          onPressed: onClose,
        ),
      ),
    ],
  );
}
}