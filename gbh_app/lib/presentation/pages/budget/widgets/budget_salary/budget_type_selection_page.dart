// lib/presentation/pages/budget/budget_type/budget_type_selection_page.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/budget/budget_type_model.dart';
import 'package:marshmellow/data/models/my/user_detail_info.dart';
import 'package:marshmellow/presentation/viewmodels/budget/budget_type_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_info_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';
import 'package:marshmellow/router/routes/budget_routes.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_salary/budget_creation_page.dart'; // Add this import

class BudgetTypeSelectionPage extends ConsumerWidget {
  const BudgetTypeSelectionPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(budgetTypeProvider);
    final viewModel = ref.read(budgetTypeProvider.notifier);
    final userInfoState = ref.watch(userInfoProvider);
    
    // 로딩 중이거나 분석 결과가 없는 경우 로딩 표시
    if (state.isLoading || state.analysisResult == null) {
      return Scaffold(
        appBar: CustomAppbar(title: '예산 유형 선택'),
        body: const Center(
          child: CustomLoadingIndicator(),
        ),
      );
    }

    // 모든 유형 정보 가져오기
    final allTypeInfos = BudgetTypeInfo.getAllTypeInfos();

    // 사용자 월급 정보 가져오기
    int salary = userInfoState is UserDetailInfo
        ? (userInfoState as UserDetailInfo).salaryAmount ?? 3000000
        : 3000000;

    return Scaffold(
      appBar: CustomAppbar(
        title: '예산 유형 선택',
        actions: [
          // 도움말 아이콘
          IconButton(
            icon: const Icon(Icons.help_outline),
            onPressed: () {
              _showHelpDialog(context);
            },
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const SizedBox(height: 20),
                  const Text(
                    '이달의 소비 유형을 선택하시면\n자동 예산 분배 해드릴게요!',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w300,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 16),
                  
                  // 가로 스크롤 유형 선택
                  Expanded(
                    child: _buildHorizontalTypeCards(
                      context,
                      allTypeInfos,
                      state.selectedType,
                      viewModel,
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          // 하단 버튼
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Button(
              text: '선택 완료',
              width: 270,
              textStyle: TextStyle(color: AppColors.whiteLight, fontWeight: FontWeight.w300),
              isLoading: state.isSavingSelection,
              onPressed: state.selectedType == null
                  ? null // 선택이 없으면 비활성화
                  : () async {
                      // 선택한 유형 정보 저장
                      final success = await viewModel.saveSelectedType();
                      if (success && context.mounted) {
                        // 성공 시 예산 생성 페이지로 이동

                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (context) => BudgetCreationPage(
                              selectedType: state.selectedType!,
                            )
                          )
                        );
                      }
                    },
            ),
          ),
        ],
      ),
    );
  }

  // 가로 스크롤 유형 카드 위젯
  Widget _buildHorizontalTypeCards(
    BuildContext context,
    List<BudgetTypeInfo> typeInfos,
    String? selectedType,
    BudgetTypeViewModel viewModel,
  ) {
    // 디바이스 너비 구하기
    final screenWidth = MediaQuery.of(context).size.width;
    // 카드 너비 설정 (양쪽이 약간 잘리도록)
    // final cardWidth = screenWidth * 0.75;

    // 선택된 카드의 인덱스 찾기
    final selectedIndex = selectedType != null 
        ? typeInfos.indexWhere((info) => info.type == selectedType) 
        : 0;
    
    // PageController 생성
    final PageController pageController = PageController(
      // 선택된 카드가 중앙에 오도록 초기 페이지 설정
      initialPage: selectedIndex >= 0 ? selectedIndex : 0,
      // viewportFraction 값을 줄여서 양쪽에 이전/다음 카드가 더 많이 보이도록 함
      viewportFraction: 0.7,
    );
    
    return Column(
      children: [
        // 카드 페이지뷰
        Expanded(
          child: LayoutBuilder(
            builder: (context, constraints) {
              return Container(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: PageView.builder(
                  controller: PageController(
              
                    // 초기 페이지
                    initialPage: 4,
                    // 양쪽이 살짝 보이도록 설정
                    viewportFraction: 0.80,
                  ),
                  padEnds: true, // 양쪽 끝 패딩 제거
                  itemCount: typeInfos.length,
                  itemBuilder: (context, index) {
                    final typeInfo = typeInfos[index];
                    final isSelected = selectedType == typeInfo.type;
                    
                    return AnimatedScale(
                      scale: isSelected ? 1.0 : 0.95,
                      duration: const Duration(milliseconds: 200),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 8.0),
                        child: _buildTypeCard(
                          context,
                          typeInfo,
                          isSelected,
                          () => viewModel.selectBudgetType(typeInfo.type),
                        ),
                      ),
                    );
                  },
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  // 유형 카드 위젯
  Widget _buildTypeCard(
    BuildContext context,
    BudgetTypeInfo typeInfo,
    bool isSelected,
    VoidCallback onSelect,
  ) {
    return GestureDetector(
      onTap: onSelect,
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected ? typeInfo.color : Colors.grey.withOpacity(0.3),
            width: isSelected ? 2 : 1,
          ),
          boxShadow: isSelected
              ? [
                  BoxShadow(
                    color: typeInfo.color.withOpacity(0.2),
                    spreadRadius: 1,
                    blurRadius: 5,
                    offset: const Offset(0, 2),
                  ),
                ]
              : null,
        ),
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // 선택 표시
            if (isSelected)
              Align(
                alignment: Alignment.topRight,
                child: Container(
                  padding: const EdgeInsets.all(4),
                  decoration: BoxDecoration(
                    color: typeInfo.color,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.check,
                    color: Colors.white,
                    size: 16,
                  ),
                ),
              ),
            
            // 아이콘 이미지
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: typeInfo.color.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(
                _getIconForType(typeInfo.type),
                color: typeInfo.color,
                size: 40,
              ),
            ),
            
            const SizedBox(height: 20),
            
            // 유형 이름
            Text(
              typeInfo.selectionName,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 18,
              ),
              textAlign: TextAlign.center,
            ),
            
            const SizedBox(height: 12),
            
            // 간단한 설명
            Text(
              _getShortDescription(typeInfo.type),
              style: const TextStyle(
                fontSize: 14,
                color: Colors.grey,
              ),
              textAlign: TextAlign.center,
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
            ),
            
            // 특성 표시
            const SizedBox(height: 20),
            _buildTypeTraits(typeInfo),
          ],
        ),
      ),
    );
  }

  // 유형별 특성 표시
  Widget _buildTypeTraits(BudgetTypeInfo typeInfo) {
    return Wrap(
      alignment: WrapAlignment.center,
      spacing: 8,
      runSpacing: 8,
      children: _getTypeTraits(typeInfo.type).map((trait) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: typeInfo.color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Text(
            trait,
            style: TextStyle(
              fontSize: 12,
              color: typeInfo.color,
              fontWeight: FontWeight.w500,
            ),
          ),
        );
      }).toList(),
    );
  }

  // 유형별 특성 목록
  List<String> _getTypeTraits(String type) {
    switch (type) {
      case '식비/외식':
        return ['먹방러버', '맛집탐방', '미식가'];
      case '교통/자동차':
        return ['효율적', '이동중심', '바쁜일상'];
      case '편의점/마트':
        return ['생활중심', '실용적', '알뜰살림'];
      case '금융':
        return ['계획형', '투자중심', '미래준비'];
      case '여가비':
        return ['문화생활', '취미활동', '힐링'];
      case '커피/디저트':
        return ['달콤한휴식', '카페러버', '소확행'];
      case '쇼핑':
        return ['자기계발', '트렌디', '자기보상'];
      case '비상금':
        return ['준비성', '계획형', '안정추구'];
      case '평균':
        return ['균형잡힘', '합리적', '중도형'];
      default:
        return ['맞춤형', '개성있는', '유니크'];
    }
  }

  // 유형별 아이콘 반환
  IconData _getIconForType(String type) {
    switch (type) {
      case '식비/외식':
        return Icons.restaurant;
      case '교통/자동차':
        return Icons.directions_car;
      case '편의점/마트':
        return Icons.shopping_basket;
      case '금융':
        return Icons.account_balance;
      case '여가비':
        return Icons.movie;
      case '커피/디저트':
        return Icons.local_cafe;
      case '쇼핑':
        return Icons.shopping_bag;
      case '비상금':
        return Icons.savings;
      case '평균':
        return Icons.balance;
      default:
        return Icons.category;
    }
  }

  // 유형별 짧은 설명
  String _getShortDescription(String type) {
    switch (type) {
      case '식비/외식':
        return '맛있는 음식에 투자하는 미식가\n당신은 음식에 가치를 둡니다';
      case '교통/자동차':
        return '효율적인 이동이 중요한 바쁜 일상\n당신은 언제나 이동중입니다';
      case '편의점/마트':
        return '생활에 필요한 것은 다 있다\n당신은 실용적인 소비를 합니다';
      case '금융':
        return '미래를 준비하는 투자자\n당신은 계획적인 소비를 추구합니다';
      case '여가비':
        return '문화생활로 삶의 질을 높이는 방법\n당신은 취미와 여가에 투자합니다';
      case '커피/디저트':
        return '달콤한 휴식과 함께하는 여유\n당신은 작은 행복을 즐깁니다';
      case '쇼핑':
        return '나를 위한 투자와 보상\n당신은 자신에게 관대합니다';
      case '비상금':
        return '언제 어디서나 준비되어 있는 여유\n당신은 미래를 대비합니다';
      case '평균':
        return '균형 잡힌 지출로 합리적인 소비\n당신은 균형을 중요시합니다';
      default:
        return '나만의 소비 스타일\n당신은 독특한 소비패턴을 가졌습니다';
    }
  }

  // 도움말 다이얼로그
  void _showHelpDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(  // 모서리 둥글기 조정
          borderRadius: BorderRadius.circular(15),
        ),
        title: Text('예산 유형이란?', style: AppTextStyles.appBar),
        content: SingleChildScrollView(
          child: Text(
            '예산 유형은 당신의 소비 패턴을 분석하여 가장 잘 맞는 예산 배분 방식을 추천해드립니다.\n\n'
            '각 유형별로 카테고리 간 예산 비율이 다르게 설정되어, 당신의 생활 패턴에 맞는 예산 계획을 자동으로 생성해 드립니다.\n\n',
            style: AppTextStyles.bodyMediumLight.copyWith(fontWeight: FontWeight.w300),
          ),
        ),
        contentPadding: EdgeInsets.fromLTRB(24, 20, 24, 24),  // 내용 패딩 조정
        titlePadding: EdgeInsets.fromLTRB(24, 24, 24, 8),  // 제목 패딩 조정
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('확인'),
          ),
        ],
      ),
    );
  }
}

// 인디케이터 위젯
class _TypeCarouselIndicator extends StatelessWidget {
  final int itemCount;
  final int currentPage;

  const _TypeCarouselIndicator({
    required this.itemCount,
    required this.currentPage,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: List.generate(
        itemCount,
        (index) => Container(
          width: index == currentPage ? 12 : 8,
          height: index == currentPage ? 12 : 8,
          margin: const EdgeInsets.symmetric(horizontal: 4),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: index == currentPage
                ? Theme.of(context).primaryColor
                : Colors.grey.withOpacity(0.3),
          ),
        ),
      ),
    );
  }
}