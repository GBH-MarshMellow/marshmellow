import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/router/routes/cookie_routes.dart';

class PortfolioCard extends StatelessWidget {
  const PortfolioCard({super.key});

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.of(context).size.height;

    return GestureDetector(
      onTap: () => context.push(CookieRoutes.getPortfolioPath()),
      child: Container(
        height: screenHeight * 0.25,
        decoration: BoxDecoration(
          color: AppColors.bluePrimary,
          borderRadius: BorderRadius.circular(15),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2), // 그림자 색상 및 투명도
              spreadRadius: 0.5, // 그림자 확산 범위
              blurRadius: 8, // 그림자 흐림 정도
              offset: Offset(0, 4), // 그림자 위치 (x, y)
            ),
          ],
        ),
        child: Stack(
          children: [
            Positioned(
              top: 20,
              left: 20,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '포트폴리오',
                    style: AppTextStyles.mainTitle,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '연봉 협상을 위한\n자료 아카이브',
                    style: AppTextStyles.bodyMedium.copyWith(
                      color: AppColors.greyPrimary,
                    ),
                  ),
                ],
              ),
            ),
            Positioned(
              top: 20,
              right: 20,
              child: SvgPicture.asset(IconPath.caretRight),
            ),
            Positioned(
              bottom: 10,
              right: 20,
              child: Image.asset(
                'assets/images/characters/char_melong.png',
                height: screenHeight * 0.175,
                fit: BoxFit.contain,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
