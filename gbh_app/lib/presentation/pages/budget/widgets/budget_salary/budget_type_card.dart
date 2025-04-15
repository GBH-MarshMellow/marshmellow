import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/budget/budget_type_model.dart';
import 'package:marshmellow/di/providers/budget/budget_type_provider.dart';
import 'package:marshmellow/presentation/viewmodels/budget/budget_type_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';

/*
  예산 유형 카드
*/
class BudgetTypeCard extends ConsumerWidget {
  final Function() onTapMoreDetails;
  
  const BudgetTypeCard({
    Key? key,
    required this.onTapMoreDetails,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {


    print('🖌️ BudgetTypePage 빌드 시작');
    final budgetTypeState = ref.watch(budgetTypeProvider);
    print('🖌️ 현재 상태: isLoading=${budgetTypeState.isLoading}, hasError=${budgetTypeState.errorMessage != null}, myType=${budgetTypeState.myBudgetType}');
    
    // 로딩 중이면 로딩 인디케이터 표시
    if (budgetTypeState.isLoading) {
      print('🖌️ 로딩 상태 표시');
      return Center(
        child: CustomLoadingIndicator(
          text: '분석 중입니다...',
          backgroundColor: Colors.white,
          opacity: 0.8,
        ),
      );
    }
    
    // 에러가 있으면 에러 메시지 표시
    if (budgetTypeState.errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text(
            '분석에 실패했습니다',
            style: AppTextStyles.bodyMediumLight.copyWith(
              color: Colors.red,
              fontSize: 16,
            ),
          ),
        ),
      );
    }
    
    // 분석 결과가 없으면 메시지 표시
    if (budgetTypeState.analysisResult == null || budgetTypeState.myBudgetType == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Text('분석 결과가 없습니다'),
        ),
      );
    }
    
    // 내 예산 유형 정보 가져오기
    final myType = budgetTypeState.myBudgetType!;
    print('🖌️ 표시할 유형: $myType');
    final myTypeInfo = BudgetTypeInfo.getTypeInfo(myType);
    print('🖌️ 유형 정보: ${myTypeInfo.typeName}');

    
    return Container(
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color:AppColors.backgroundBlack, width: 0.5),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(height: 20),

          // 유형 이름
          Text(
            '${myTypeInfo.typeName}',
            style: AppTextStyles.mainTitle.copyWith(
              fontSize: 30,
              fontWeight: FontWeight.w500,
              color: myTypeInfo.color
            ),
            textAlign: TextAlign.center,
          ),
          
          SizedBox(height: 20),

          // 이미지
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 16.0),
            child: 
            Image.asset(
                'assets/images/characters/char_jump.png',
                width: 150,
                height: 150,
              ),
          ),

          // const SizedBox(height: 10),

          // 유형 소비 비율 설명
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 10.0),
            child: Text(
              '전체 예산 중\n${(ref.read(budgetTypeProvider.notifier).getMyTypeRatio() * 100).toStringAsFixed(0)}%를 '
              '${myType}으로 사용하고 있습니다.',
              style: AppTextStyles.bodyMediumLight.copyWith(
                fontSize: 13,
                fontWeight: FontWeight.w400,
                color: myTypeInfo.color,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          
          // 유형 설명 (간략하게)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Text(
              myTypeInfo.description,
              style: AppTextStyles.bodyMediumLight.copyWith(
                fontSize: 16,
                fontWeight: FontWeight.w300,
              ),
              textAlign: TextAlign.center,
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          
          SizedBox(height: 24),
          
          // 더 자세히 보기 버튼
          Button(
            onPressed: onTapMoreDetails,
            text: '예산 유형 선택하러가기',
            textStyle :AppTextStyles.bodyMediumLight.copyWith(
                fontSize: 16,
                fontWeight: FontWeight.w200,
                color: Colors.white,
              ),
            width: 240,
            ),

        ],
      ),
    );
  }
}