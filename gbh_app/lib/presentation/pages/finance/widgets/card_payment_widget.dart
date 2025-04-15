// presentation/pages/finance/widgets/total_assets_widget.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/banner_ad_widget.dart';

class TotalAssetsWidget extends ConsumerWidget {
  final int totalAssets;
  
  const TotalAssetsWidget({
    Key? key,
    required this.totalAssets,
  }) : super(key: key);
  
  // 숫자 포맷팅 함수 (천 단위 구분)
  String formatAmount(int amount) {
    final formatter = NumberFormat('#,###');
    return formatter.format(amount);
  }
  
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isHidden = ref.watch(isFinanceHideProvider);
    
    return SizedBox(
      height: 180, // 전체 높이 조정
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 왼쪽 영역: 총자산 + 광고
          Expanded(
            flex: 3, // 왼쪽 영역 비율
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // 상단: 총 자산 부분
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 0, horizontal: 4),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // 왼쪽 텍스트 부분
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            '총 자산',
                            style: AppTextStyles.bodyMedium,
                          ),
                          // const SizedBox(height: 8),
                          Text(
                            isHidden ? '자산보기' : '${formatAmount(totalAssets + 0)}원',
                            style: isHidden 
                              ? AppTextStyles.bodyLarge 
                              : AppTextStyles.bodyLarge.copyWith(letterSpacing: -1),
                          ),
                        ],
                      ),
                      // 오른쪽 토글 스위치
                      Transform.scale(
                        scale: 0.7,  // 1보다 작으면 축소, 1보다 크면 확대
                        child: Switch(
                          value: isHidden,
                          onChanged: (value) {
                            ref.read(isFinanceHideProvider.notifier).state = value;
                          },
                          activeTrackColor: AppColors.greyPrimary,
                          inactiveTrackColor: AppColors.backgroundBlack,
                          activeColor: Colors.white,
                          inactiveThumbColor: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                // 하단: 광고 영역
                const Expanded(
                  child: Padding(
                    padding: EdgeInsets.only(top: 8, right: 8),
                    child: BannerAdWidget(),
                  ),
                ),
              ],
            ),
          ),
          
          // 오른쪽 영역: 캐릭터 이미지
          Expanded(
            flex: 2, // 오른쪽 영역 비율
            child: Container(
              height: double.infinity,
              alignment: Alignment.bottomRight,
              decoration: BoxDecoration(
                // color: AppColors.greenPrimary,
                // borderRadius: BorderRadius.all(Radius.circular(10.0)),
                // border: Border.all(width: 2.0 , color: AppColors.blackLight)
              ),
              child: Image.asset(
                'assets/images/characters/char_hat.png',
                fit: BoxFit.contain,
              ),
            ),
          ),
        ],
      ),
    );
  }
}