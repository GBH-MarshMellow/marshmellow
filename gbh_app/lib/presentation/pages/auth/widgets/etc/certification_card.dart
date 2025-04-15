import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

/*
  금융인증서 카드 - 이미지와 정확히 동일한 버전
*/
class CertificateCard extends StatelessWidget {
  final String userName;
  final String expiryDate;
  final double width;
  final double height;
  final Color backgroundColor;
  final Color textColor;
  final Color expiryLabelColor;
  final VoidCallback? onTap;
  final String? shieldSvgPath;

  const CertificateCard({
    Key? key,
    required this.userName,
    required this.expiryDate,
    this.width = 220,
    this.height = 330,
    this.backgroundColor = AppColors.yellowPrimary,
    this.textColor = AppColors.backgroundBlack,
    this.expiryLabelColor = AppColors.whiteLight,
    this.onTap,
    this.shieldSvgPath = 'assets/icons/etc/shield.svg',
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // 고정된 비율로 카드 생성
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: width,
        height: height,
        decoration: BoxDecoration(
          color: backgroundColor,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Stack(
          children: [
            // SVG 쉴드 배경 - 전체 카드를 덮음
            Positioned.fill(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: SvgPicture.asset(
                  shieldSvgPath!,
                  fit: BoxFit.contain,
                ),
              ),
            ),
              
            // 텍스트 레이어 (SVG 위에 표시)
            Positioned.fill(
              child: Padding(
                padding: const EdgeInsets.all(25.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // 상단 텍스트
                    Text(
                      'MM 인증서',
                      style: AppTextStyles.bodyMedium.copyWith(
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    
                    // 하단 텍스트 영역
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          userName,
                          style: AppTextStyles.bodyLargeLight.copyWith(
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            Text(
                              expiryDate,
                              style: AppTextStyles.bodySmall
                            ),
                            const SizedBox(width: 4),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1,),
                              decoration: BoxDecoration(
                                color: AppColors.whiteLight,
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Text(
                                '정상',
                                style: AppTextStyles.bodySmall.copyWith(
                                  color: AppColors.yellowPrimary,
                                  fontSize: 10,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/*
  mm인증서 카드 ver.small
*/
class SmallCertificateCard extends StatelessWidget {
  final String userName;
  final String expiryDate;
  final double width;
  final double height;
  final Color backgroundColor;
  final Color textColor;
  final Color expiryLabelColor;
  final VoidCallback? onTap;
  final String? shieldSvgPath;

  const SmallCertificateCard({
    Key? key,
    required this.userName,
    required this.expiryDate,
    this.width = 160,
    this.height = 240,
    this.backgroundColor = AppColors.yellowPrimary,
    this.textColor = AppColors.backgroundBlack,
    this.expiryLabelColor = AppColors.whiteLight,
    this.onTap,
    this.shieldSvgPath = 'assets/icons/etc/shield.svg',
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // 고정된 비율로 카드 생성
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: width,
        height: height,
        decoration: BoxDecoration(
          color: backgroundColor,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Stack(
          children: [
            // SVG 쉴드 배경 - 전체 카드를 덮음
            Positioned.fill(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: SvgPicture.asset(
                  shieldSvgPath!,
                  fit: BoxFit.contain,
                ),
              ),
            ),
              
            // 텍스트 레이어 (SVG 위에 표시)
            Positioned.fill(
              child: Padding(
                padding: const EdgeInsets.all(25.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // 상단 텍스트
                    Text(
                      'MM 인증서',
                      style: AppTextStyles.bodySmall.copyWith(
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    
                    // 하단 텍스트 영역
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          userName,
                          style: AppTextStyles.bodyMediumLight
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            Text(
                              expiryDate,
                              style: AppTextStyles.bodyExtraSmall
                            ),
                            const SizedBox(width: 4),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1,),
                              decoration: BoxDecoration(
                                color: AppColors.whiteLight,
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Text(
                                '정상',
                                style: AppTextStyles.bodyExtraSmall.copyWith(
                                  color: AppColors.yellowPrimary,
                                  fontSize: 8,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}