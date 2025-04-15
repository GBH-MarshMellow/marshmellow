// lib/presentation/pages/budget/budget_type/budget_type_page.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/data/models/budget/budget_type_model.dart';
import 'package:marshmellow/presentation/viewmodels/budget/budget_type_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';
import 'package:marshmellow/router/routes/budget_routes.dart';

class BudgetTypePage extends ConsumerWidget {
  const BudgetTypePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    print('🖌️ BudgetTypePage 빌드 시작');
    final state = ref.watch(budgetTypeProvider);
    print('🖌️ 현재 상태: isLoading=${state.isLoading}, hasError=${state.errorMessage != null}, myType=${state.myBudgetType}');
    
    // 로딩 중이거나 분석 결과가 없는 경우 로딩 표시
    if (state.isLoading || state.analysisResult == null) {
      print('🖌️ 로딩 상태 표시');
      return Scaffold(
        appBar: CustomAppbar(title: '예산 유형 분석'),
        body: const Center(
          child: CustomLoadingIndicator(),
        ),
      );
    }

    // 에러가 있는 경우 에러 메시지 표시
    if (state.errorMessage != null) {
      print('🖌️ 에러 상태 표시: ${state.errorMessage}');
      return Scaffold(
        appBar: CustomAppbar(title: '예산 유형 분석'),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 48, color: Colors.red),
              const SizedBox(height: 16),
              Text(
                state.errorMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 16),
              ),
              const SizedBox(height: 24),
              Button(
                text: '다시 시도',
                onPressed: () => ref.read(budgetTypeProvider.notifier).analyzeBudgetType(),
              ),
            ],
          ),
        ),
      );
    }

    // 내 유형 정보
    final myType = state.myBudgetType ?? '평균';
    print('🖌️ 표시할 유형: $myType');
    final myTypeInfo = BudgetTypeInfo.getTypeInfo(myType);
    print('🖌️ 유형 정보: ${myTypeInfo.typeName}');

    print('🖌️ 정상 UI 렌더링');
    return Scaffold(
      appBar: CustomAppbar(title: '예산 유형 분석'),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 현재 나의 유형
              _buildMyTypeSection(context, myTypeInfo),
              
              const SizedBox(height: 32),
              
              // 다른 유형 보기 버튼
              Button(
                text: '다른 유형으로 설정하기',
                onPressed: () {
                  context.push(SignupRoutes.getBudgetTypeSelectionPath());
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  // 내 유형 섹션 위젯
  Widget _buildMyTypeSection(BuildContext context, BudgetTypeInfo typeInfo) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            spreadRadius: 1,
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text(
            '지독한 ${typeInfo.typeName}!',
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
              height: 1.5,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 24),
          
          // 캐릭터 이미지 (실제 앱에서는 적절한 이미지 경로로 변경 필요)
          Image.asset(
            'assets/images/budget/mascot.png', // 기본 이미지 (실제 개발시 변경 필요)
            height: 180,
            fit: BoxFit.contain,
          ),
          
          const SizedBox(height: 24),
          
          // 유형 설명 텍스트
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Text(
              typeInfo.description,
              style: const TextStyle(
                fontSize: 16,
                height: 1.4,
                color: Colors.black87,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }
}