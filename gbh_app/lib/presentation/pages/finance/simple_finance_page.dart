// presentation/pages/finance/finance_page.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/config/app_config.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/finance_analytics_widget.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/loading/loading_manager.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';

import 'package:marshmellow/presentation/pages/finance/widgets/total_assets_widget.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/simple/sliding_assets_widget.dart';

// appbar 간편버튼
import 'package:marshmellow/presentation/pages/finance/widgets/simple_toggle_button_widget.dart';

class SimpleFinancePage extends ConsumerStatefulWidget  {
  const SimpleFinancePage({super.key});

  @override
  _SimpleFinancePageState createState() => _SimpleFinancePageState();
}

class _SimpleFinancePageState extends ConsumerState<SimpleFinancePage> {
  // 상태 변수를 여기에 선언
  bool _showAnalyticsWidget = true;

  @override
  void initState() {
    super.initState();
    // 위젯 초기화 후 데이터 로드 요청
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(financeViewModelProvider.notifier).fetchAssetInfo();
    });
  }

  @override
  Widget build(BuildContext context) {
    // StateNotifierProvider로 상태 가져오기
    final financeState = ref.watch(financeViewModelProvider);

    // 로딩 상태에 따라 LoadingManager 처리
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (financeState.isLoading) {
        LoadingManager.show(context, text: '자산 정보를 불러오는 중...');
      } else {
        LoadingManager.hide();
      }
    });

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: CustomAppbar(
        title: 'my little 자산',
        backgroundColor: AppColors.background,
        actions: [
          if (AppConfig.isDevelopment())
            IconButton(
              icon: const Icon(Icons.bug_report),
              onPressed: () {
                context.push(FinanceRoutes.getTestPath());
              },
              tooltip: '테스트 페이지로 이동',
            ),
          const SimpleToggleButton(isSimplePage: true),
        ],
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Expanded(
              child: _buildContent(financeState),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildContent(FinanceState state) {
    // 로딩 중
    if (state.isLoading && state.assetData == null) {
      return Container(); // LoadingManager가 오버레이로 표시됨
    }
    
    // 에러 발생
    if (state.error != null && state.assetData == null) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline, size: 48, color: Colors.red),
          const SizedBox(height: 10),
          Text('에러 발생: ${state.error}'),
          ElevatedButton(
            onPressed: () => ref.read(financeViewModelProvider.notifier).refreshAssetInfo(),
            child: const Text('다시 시도'),
          ),
        ],
      );
    }
    
    // 데이터 없음
    if (state.assetData == null) {
      return const Center(child: Text('자산 데이터가 없습니다'));
    }
    
    // 성공적으로 데이터를 받았을 때
    final viewModel = ref.read(financeViewModelProvider.notifier);
    final totalAssets = viewModel.calculateTotalAssets();
    
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 총 자산 정보
          TotalAssetsWidget(totalAssets: totalAssets),
          
          const SizedBox(height: 24),
          
          // 새로운 슬라이딩 자산 위젯 사용
          SlidingAssetsWidget(data: state.assetData!),

          const SizedBox(height: 24),

          // if (_showAnalyticsWidget)
          //   FinanceAnalyticsWidget(
          //     onClose: () {
          //       setState(() {
          //         _showAnalyticsWidget = false;
          //       });
          //     },
          //   ),
        ],
      ),
    );
  }
}