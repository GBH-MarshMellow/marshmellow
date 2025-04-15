import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/analysis/finance_loading_animation.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/analysis/finance_result_detail.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_analysis_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';

class FinanceAnalysisPage extends ConsumerStatefulWidget {
  const FinanceAnalysisPage({Key? key}) : super(key: key);
  

  @override
  ConsumerState<FinanceAnalysisPage> createState() => _FinanceAnalysisPageState();
}

class _FinanceAnalysisPageState extends ConsumerState<FinanceAnalysisPage> {
  Timer? _autoShowDetailTimer;

  @override
  void initState() {
    super.initState();
    // 자동 시작 코드 제거
    // 페이지 진입 시 자산 데이터만 미리 로드 (선택사항)
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _preloadAssetData();
    });
  }
  // 자산 데이터 미리 로드하는 메서드 추가 (initState 아래)
  void _preloadAssetData() async {
    final financeState = ref.read(financeViewModelProvider);
    if (financeState.assetData == null) {
      await ref.read(financeViewModelProvider.notifier).fetchAssetInfo();
    }
  }

  @override
  void dispose() {
    // 타이머 정리
    _autoShowDetailTimer?.cancel();
    super.dispose();
  }

  // 분석 시작 메서드
  void _startAnalysis() async {
    final financeState = ref.read(financeViewModelProvider);
    // 자산 데이터가 없으면 먼저 로드
    if (financeState.assetData == null) {
      await ref.read(financeViewModelProvider.notifier).fetchAssetInfo();
    }
    
    // 분석 시작
    await ref.read(financeAnalysisViewModelProvider.notifier).startAnalysis();
    
    // 분석 결과가 나오면 3초 후 자동으로 상세 정보 표시
    // final currentState = ref.read(financeAnalysisViewModelProvider);
    // if (currentState.status == AnalysisStatus.shortResult) {
    //   _autoShowDetailTimer = Timer(const Duration(seconds: 3), () {
    //     ref.read(financeAnalysisViewModelProvider.notifier).showDetailResult();
    //   });
    // }
  }

  @override
  Widget build(BuildContext context) {
    // 분석 상태 감시
    final analysisState = ref.watch(financeAnalysisViewModelProvider);
    
    return Scaffold(
      appBar: CustomAppbar(title: '자산유형분석'),
      body: SafeArea(
        child: Padding(
          padding: EdgeInsets.fromLTRB(10.0, 10.0, 10.0, 0.0),
          child: _buildContent(analysisState),
        ),
      ),
    );
  }

  // _buildContent 메서드 수정 (switch 문 부분)
  Widget _buildContent(FinanceAnalysisState state) {
    // 에러 발생 시
    if (state.error != null) {
      return _buildErrorContent(state.error!);
    }
    
    // 상태에 따른 UI 분기
    switch (state.status) {
      case AnalysisStatus.initial:
      case AnalysisStatus.ready:
        // 준비 상태 - 정지된 애니메이션 표시
        return Center(
          child: FinanceLoadingAnimation(
            state: LoadingAnimationState.ready,
            onStartPressed: _startAnalysis,
            resultTypeId: state.selectedType?.id, // 선택된 타입 ID 전달
          ),
        );
        
      case AnalysisStatus.analyzing:
        // 분석 중 - 동작하는 애니메이션 표시 (같은 위젯, 다른 상태)
        return Center(
          child: FinanceLoadingAnimation(
            state: LoadingAnimationState.analyzing,
            onStartPressed: _startAnalysis, // 여기서는 사용되지 않지만 필요한 매개변수
            resultTypeId: state.selectedType?.id, // 선택된 타입 ID 전달
          ),
        );

      case AnalysisStatus.detailResult:
        // 상세 결과 - 상세 정보 표시
        return _buildDetailResult(state);
    }
  }

  // 상세 결과 UI
  Widget _buildDetailResult(FinanceAnalysisState state) {
    if (state.selectedType == null) {
      return const Center(
        child: Text('분석 결과를 찾을 수 없습니다.'),
      );
    }
    
    return Center(
      child: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // 상세 결과 위젯
            FinanceResultDetail(
              financeType: state.selectedType!,
            ),
            const SizedBox(height: 16),
            // 다시 분석하기 버튼
            InkWell(
              onTap: () {
                // 분석 초기화 후 다시 시작
                ref.read(financeAnalysisViewModelProvider.notifier).resetAnalysis();
                // _startAnalysis();
              },
              child: Container(
                width: MediaQuery.of(context).size.width,
                height: 80, // 높이 조정
                decoration: BoxDecoration(
                  // color: AppColors.backgroundBlack, // 버튼 색상 (적절히 변경하세요)
                  image: DecorationImage(
                    image: AssetImage('assets/images/finance/finance_background_${state.selectedType!.id}.png'),
                    fit: BoxFit.cover, // 이미지가 컨테이너를 꽉 채우도록 설정
                  ),
                  borderRadius: BorderRadius.only(
                    topLeft: Radius.circular(160), 
                    topRight: Radius.circular(160),
                  ), // 상단만 둥글게
                ),
                alignment: Alignment.topCenter, // 텍스트를 상단 중앙에 배치
                padding: EdgeInsets.only(top: 20), // 텍스트 위치 조정
                child: Text(
                  '다시 분석하기',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
            // SizedBox(height: 10),
          ],
        ),
      ),
    );
  }

  // 에러 UI
  Widget _buildErrorContent(String error) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(
            Icons.error_outline,
            color: AppColors.buttonDelete,
            size: 48,
          ),
          const SizedBox(height: 16),
          Text(
            '오류가 발생했습니다',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: AppColors.buttonDelete,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            error,
            style: const TextStyle(
              fontSize: 14,
              color: Colors.grey,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 24),
          ElevatedButton(
            onPressed: () {
              // 다시 시도
              ref.read(financeAnalysisViewModelProvider.notifier).resetAnalysis();
              _startAnalysis();
            },
            child: const Text('다시 시도'),
          ),
        ],
      ),
    );
  }
}