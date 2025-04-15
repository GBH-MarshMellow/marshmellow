// presentation/pages/finance/widgets/banner_ad_widget.dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';

class FinanceAnalyticsWidget extends StatelessWidget {
  final VoidCallback? onClose; // 닫기 콜백 추가
  
  const FinanceAnalyticsWidget({
    Key? key, 
    this.onClose,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
  return Stack(
    children: [
      Container(
        width: MediaQuery.of(context).size.width * 0.9,
        padding: const EdgeInsets.all(32),
        decoration: BoxDecoration(
          color: AppColors.bluePrimary,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text(
              '오직 MM에서만 만날 수 있는',
              style: AppTextStyles.bodyMedium
            ),
            const SizedBox(height: 8),
            Text(
              '자산유형분석',
              style: AppTextStyles.bodyExtraLarge.copyWith(
                letterSpacing: -2.0, // 자간 줄이기 (음수 값일수록 더 좁아짐)
                fontWeight: FontWeight.bold, // 글자 굵기 올리기
              )
            ),
            const SizedBox(height: 8),
            const Text(
              '나의 유형을 테스트 해보세요',
              style: AppTextStyles.bodyMedium
            ),
            const SizedBox(height: 16),
            GestureDetector(
              onTap: () {
                context.push(FinanceRoutes.getAnalysisPath());
              },
              child: Container(
                padding: EdgeInsets.fromLTRB(40.0, 14.0, 40.0, 14.0),
                decoration: BoxDecoration(
                  color: AppColors.buttonBlack,
                  borderRadius: BorderRadius.all(Radius.circular(6.0))
                ),
                child: Text('분석하기', style: AppTextStyles.bodyMedium
                  .copyWith(color: AppColors.blueLight),),
              ),
            ),
          ],
        ),
      ),
      Positioned(
        top: -4,
        right: -4,
        child: IconButton(
          icon: SvgPicture.asset(IconPath.exitgray,
            width: 24,
            height: 24,
          ),
          onPressed: onClose,
        ),
      ),
    ],
  );
}
}