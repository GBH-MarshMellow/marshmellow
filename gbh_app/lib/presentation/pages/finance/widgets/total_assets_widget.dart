// presentation/pages/finance/widgets/total_assets_widget.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/banner_ad_widget.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';

class TotalAssetsWidget extends ConsumerStatefulWidget {
  final int totalAssets;
  final ScrollController? scrollController; // 스크롤 컨트롤러 추가
  
  const TotalAssetsWidget({
    Key? key,
    required this.totalAssets,
    this.scrollController, // 스크롤 컨트롤러 파라미터 추가
  }) : super(key: key);
  
  @override
  ConsumerState<TotalAssetsWidget> createState() => _TotalAssetsWidgetState();
}

class _TotalAssetsWidgetState extends ConsumerState<TotalAssetsWidget> {
  bool _showBannerAdWidget = true; // 초기값 설정
  
  // 숫자 포맷팅 함수 (천 단위 구분)
  String formatAmount(int amount) {
    final formatter = NumberFormat('#,###');
    return formatter.format(amount);
  }
  
  @override
  Widget build(BuildContext context) {
  final isHidden = ref.watch(isFinanceHideProvider);
  
  return SizedBox(
    height: 180, // 전체 높이 조정
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // 왼쪽 영역: 총자산 + 광고
        Expanded(
          flex: 3, // 왼쪽 영역 비율
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 300), // 애니메이션 지속 시간
            child: _showBannerAdWidget 
              // 광고가 보이는 경우 - 기존 레이아웃
              ? Column(
                  key: const ValueKey('with_banner'), // 키 추가
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
                              Text(
                                isHidden ? '자산보기' : '${formatAmount(widget.totalAssets + 0)}원',
                                style: isHidden 
                                  ? AppTextStyles.bodyLarge 
                                  : AppTextStyles.bodyLarge.copyWith(letterSpacing: -1),
                              ),
                            ],
                          ),
                          // 오른쪽 토글 스위치
                          Transform.scale(
                            scale: 0.7,
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
                    Expanded(
                      child: Padding(
                        padding: EdgeInsets.only(top: 8, right: 8),
                        child: BannerAdWidget(
                          onClose: () {
                            setState(() {
                              _showBannerAdWidget = false;
                            });
                          },
                          scrollController: widget.scrollController, // 스크롤 컨트롤러 전달
                        ),
                      ),
                    ),
                  ],
                )
              // 광고가 숨겨진 경우 - 중앙 정렬 레이아웃
              : Center(
                  key: const ValueKey('without_banner'), // 키 추가
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 4),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        // 왼쪽 텍스트 부분
                        Column(
                          mainAxisSize: MainAxisSize.min, // 내용에 맞게 크기 조정
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              '총 자산',
                              style: AppTextStyles.bodyMedium,
                            ),
                            Text(
                              isHidden ? '자산보기' : '${formatAmount(widget.totalAssets + 0)}원',
                              style: isHidden 
                                ? AppTextStyles.bodyLarge 
                                : AppTextStyles.bodyLarge.copyWith(letterSpacing: -1),
                            ),
                          ],
                        ),
                        // 오른쪽 토글 스위치
                        Transform.scale(
                          scale: 0.7,
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
                ),
          ),
        ),
        
        // 오른쪽 영역: 캐릭터 이미지
Expanded(
  flex: 2, // 오른쪽 영역 비율
  child: Container(
    height: double.infinity,
    alignment: Alignment.bottomRight,
    child: Stack(
      children: [
        GestureDetector(
          onTap: () {
            context.push(FinanceRoutes.getAnalysisPath());
          },
          child: Image.asset(
            'assets/images/characters/char_hat.png',
            fit: BoxFit.contain,
          ),
        ),
      ],
    ),
  ),
),
      ],
    ),
  );
}
}