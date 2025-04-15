import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/data/models/wishlist/wish_model.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_bubble_chart.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/wish/wish_detail_modal.dart';
import 'package:marshmellow/presentation/viewmodels/budget/budget_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/wishlist/wish_provider.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';
import 'package:shared_preferences/shared_preferences.dart';

class BudgetPage extends ConsumerStatefulWidget {
  const BudgetPage({super.key});

  @override
  ConsumerState<BudgetPage> createState() => _BudgetPageState();
}

class _BudgetPageState extends ConsumerState<BudgetPage> with WidgetsBindingObserver {

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    // 화면이 처음 로드될 때 
    Future.microtask(() {
      // 위시리스트 데이터 가져오기
      ref.read(wishProvider.notifier).fetchCurrentWish();
      // 모든 예산 목록 가져오기
      ref.read(budgetProvider.notifier).fetchBudgets();
      // 오늘의 예산 데이터 명시적으로 새로고침
      ref.read(budgetProvider.notifier).fetchDailyBudget();
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      // 앱이 포그라운드로 돌아왔을 때 데이터 새로고침
      ref.read(budgetProvider.notifier).fetchDailyBudget();
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(budgetProvider);
    // 위시 상태 가져오기
    final wishState = ref.watch(wishProvider);

    if (state.isLoading) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        )
      );
    }

    if (state.budgets.isEmpty) {
      return Scaffold(
        // backgroundColor: Colors.white,
        appBar: CustomAppbar(
        title: '예산',
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset(
                'assets/images/characters/char_angry_notebook.png', 
                width: 180,
                height: 180,
              ),
              const SizedBox(height: 16),
              const Text(
                '앗! 예산 데이터를 불러올 수 없어요',
                style: AppTextStyles.bodyMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      );
    }

    final selectedBudget = state.selectedBudget;
    if (selectedBudget == null) {
      return const Scaffold(
        body: Center(
          child: Text('선택된 예산이 없습니다')
        ),
      );
    }

    if (state.errorMessage != null) {
      return Scaffold(
        backgroundColor: Colors.white,
        appBar: CustomAppbar(
          title: '예산',
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset(
                'assets/images/characters/char_angry_notebook.png', 
                width: 180,
                height: 180,
              ),
              const SizedBox(height: 16),
              const Text(
                '앗! 예산 데이터를 불러올 수 없어요',
                style: AppTextStyles.bodyMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              ElevatedButton(
                onPressed: () {
                  ref.read(budgetProvider.notifier).fetchBudgets();
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.bluePrimary,
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                ),
                child: const Text(
                  '다시 시도하기',
                  style: TextStyle(color: Colors.white),
                ),
              ),
            ],
          ),
        ),
      );
    }

    final categories = selectedBudget.budgetCategoryList;

    // 금액 포맷팅 (천 단위 쉼표)
    String formmatedTotalBudget = selectedBudget.budgetAmount.toString().replaceAllMapped(
      RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
      (Match m) => '${m[1]},'
    );

    // 현재 날짜
    final now = DateTime.now();
    // 선택된 예산 시작일과 종료일 파싱
    final startDate = DateTime.parse(selectedBudget.startDate);
    final endDate = DateTime.parse(selectedBudget.endDate);
    // 현재 날짜가 예산 기간 내에 있는지 확인
    final isCurrentBudget = now.isAfter(startDate.subtract(const Duration(days: 1))) &&
                            now.isBefore(endDate.subtract(const Duration(days: 1)));

    final startMonth = startDate.month;
    // final appTitle = '$startMonth원 교통 자동차';
    final appTitle = '$startMonth월 예산';

    return Scaffold(
      appBar: CustomAppbar(
        title: appTitle,
        actions: [
          IconButton(
            icon: SvgPicture.asset(IconPath.analysis),
            onPressed: () {
              context.go('/budget/detail/${selectedBudget.budgetPk}');
            },
          ),
        ],),
      body: Column(
        children: [
          // 예산 정보 요약
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 22.0, vertical: 8.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // 좌측 : 전체 남은 예산
                RichText(
                  text: TextSpan(
                    children: [
                      TextSpan(
                        text: formmatedTotalBudget,
                        style: AppTextStyles.congratulation
                      ),
                      const TextSpan(
                        text: ' 원',
                        style: AppTextStyles.bodyMedium
                      ),
                    ],
                  ),
                ),
                // 우측: 총액과 하루 예산
                if (isCurrentBudget && state.dailyBudget != null) _buildCurrentBudgetInfo(state)
              ],
            ),
          ),

          // 날짜 범위 선택기
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                IconButton(
                  onPressed: state.currentBudgetIndex < state.budgets.length - 1
                    ? () {
                        ref.read(budgetProvider.notifier).navigateToNextBudget();
                      }
                    : null,
                  icon: Icon(
                    Icons.chevron_left,
                    color: state.currentBudgetIndex < state.budgets.length - 1
                      ? AppColors.backgroundBlack
                      : AppColors.whiteDark
                    ),
                ),
                Text(
                  state.dateRangeText,
                  style: AppTextStyles.bodySmall,
                ),
                IconButton(
                  onPressed: state.currentBudgetIndex > 0 
                    ? () {
                        ref.read(budgetProvider.notifier).navigateToPreviousBudget();
                      }
                    : null,
                  icon: Icon(
                          Icons.chevron_right,
                          color: state.currentBudgetIndex > 0 
                            ? AppColors.backgroundBlack 
                            : AppColors.whiteDark, // 조건에 따라 색상 변경
                        ),
                ),
              ],
            )
          ),

          SizedBox(height: 10),

          // 메인 버블 차트 
          Expanded(
            flex: 3,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22.0),
              child: categories.isNotEmpty
              ? BudgetBubblechart(categories: categories)
              : const Center(child: Text('등록된 예산이 없습니다')),
            ),
          ),

          // 위시 섹션
          Expanded(
            flex: 2,
            child: Padding(
              padding: const EdgeInsets.all(22.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '위시',
                    style: AppTextStyles.appBar,
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: wishState.isLoading
                      ? const Center(child: CircularProgressIndicator())
                      : wishState.errorMessage != null
                        ? Center(child: Text('위시를 불러오는 중 오류가 발생했습니다.'))
                        : wishState.currentWish == null
                          ? _buildAddWishButton(context)
                          : _buildWishItem(context, wishState.currentWish!),
                  )
                ],
              )
            ),
          )
        ],
      )
    );
  }

  // 현재 진행 중인 예산 정보 위젯
  Widget _buildCurrentBudgetInfo(BudgetState state) {
    final remainingBudget = state.dailyBudget!.remainBudgetAmount;
    final dailyBudget = state.dailyBudget!.dailyBudgetAmount;

    // 금액 포맷팅
    String formattedRemainingBudget = remainingBudget.toString().replaceAllMapped(
      RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
      (Match m) => '${m[1]},'
    );

    String formattedDailyBudget = dailyBudget.toString().replaceAllMapped(
      RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
      (Match m) => '${m[1]},'
    );

    return Column(
      crossAxisAlignment: CrossAxisAlignment.end,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '남은 예산 $formattedRemainingBudget 원',
          style: AppTextStyles.bodyExtraSmall, 
        ),
        const SizedBox(height: 2),
        Text(
          '오늘의 예산 $formattedDailyBudget 원',
          style: AppTextStyles.bodyExtraSmall,
        )
      ],
    );
  }
  
  /// 위시 없을 때 추가 버튼
  Widget _buildAddWishButton(BuildContext context) {
    return InkWell(
      onTap: () {
        _showWishListModal(context, ref);
      },
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 8.0),
        decoration: BoxDecoration(
          border: Border.all(color: AppColors.backgroundBlack),
          borderRadius: BorderRadius.circular(8),
          color: Colors.white,
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 50,
                height: 50,
                child: Icon(
                  Icons.add, 
                  color: AppColors.greyLight, 
                  size: 30
                ),
              ),
              Text(
                '위시리스트 추가',
                style: AppTextStyles.bodySmall.copyWith(
                  color: AppColors.greyPrimary,
                  fontWeight: FontWeight.w300,
                ),

              )
            ],
          ),
        ),
      ),
    );
  }

  /// 위시 아이템 위젯
  Widget _buildWishItem(BuildContext context, WishDetail wish) {
    // 가격 포맷팅
    String formattedPrice = wish.productPrice.toString().replaceAllMapped(
      RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
      (Match m) => '${m[1]},'
    );

    String formattedAchievePrice = wish.achievePrice.toString().replaceAllMapped(
      RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
      (Match m) => '${m[1]},'
    );

    // 이미지 URL 처리 
    String? imageUrl;
    if (wish.productImageUrl != null) {
      if (wish.productImageUrl!.startsWith('//')) {
        // 프로토콜이 없는 URL에 https: 추가
        imageUrl = 'https:${wish.productImageUrl!}';
      } else if (!wish.productImageUrl!.startsWith('file://')) {
        // 일반 URL은 그대로 사용
        imageUrl = wish.productImageUrl;
      }
    }

    // 달성률 계산 (0~1 사이 값으로 제한)
    final progressValue = wish.achievementRate / 100;
    final clampedProgress = progressValue.clamp(0.0, 1.0);

    return InkWell(
      onTap: () {
        _showWishDetailModal(context, wish, ref);
      },
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 8.0),
        decoration: BoxDecoration(
          border: Border.all(color: AppColors.backgroundBlack),
          borderRadius: BorderRadius.circular(8),
          color: Colors.white,
        ),
        child: Padding(
          padding: const EdgeInsets.only(left: 16.0),
          child: Row(
            children: [
              // 이미지 : 원형으로 클립핑
              ClipOval(
                child: Container(
                  width: 70,
                  height: 70,
                  color: AppColors.whiteDark, // 이미지 로드 실패시 배경색
                  child: imageUrl != null
                      ? Image.network(
                          imageUrl,
                          fit: BoxFit.cover,
                          errorBuilder: (context, error, stackTrace) {
                            print('이미지 로드 실패: $imageUrl');
                            return const Icon(Icons.image_not_supported, color: Colors.grey);
                          },
                        )
                      : const Icon(Icons.image_not_supported, color: Colors.grey),
                ),
              ),
              const SizedBox(width: 12),
              // 정보 부분
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.favorite, size: 18, color: AppColors.backgroundBlack),
                          const SizedBox(width: 4),
                          Text(
                            wish.productNickname,
                            style: AppTextStyles.bodyMedium,
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(
                        wish.productName,
                        style: AppTextStyles.bodySmall,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '달성액: $formattedAchievePrice원  목표액: $formattedPrice원',
                        style: AppTextStyles.bodyExtraSmall.copyWith(color: AppColors.disabled),
                      ),
                      const SizedBox(height: 8),
                      // 진행바
                      ClipRRect(
                        borderRadius: BorderRadius.circular(30),
                        child: LinearProgressIndicator(
                          value: clampedProgress,
                          backgroundColor: AppColors.whiteDark,
                          valueColor: AlwaysStoppedAnimation<Color>(AppColors.bluePrimary),
                          minHeight: 15,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }


  /// 위시 목록 모달 표시
  void _showWishListModal(BuildContext context, WidgetRef ref) {
    showCustomModal(
      context: context,
      ref: ref,
      backgroundColor: Colors.white,
      child: const WishDetailModal(initialTab: WishListTab.pending),
    );
  }

  /// 위시 상세 모달 표시
  void _showWishDetailModal(BuildContext context, WishDetail wish, WidgetRef ref) {
    showCustomModal(
      context: context,
      ref: ref,
      backgroundColor: Colors.white,
      showDivider: false,
      child: WishDetailModal(currentWish: wish),
    );
  }
}