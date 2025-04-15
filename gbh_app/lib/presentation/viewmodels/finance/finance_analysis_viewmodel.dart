import 'dart:math';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/constants/finance_types.dart';
import 'package:marshmellow/data/models/finance/asset_response_model.dart';
import 'package:marshmellow/data/models/finance/finance_type_model.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';

// 분석 상태 열거형 (분석 진행 단계)
enum AnalysisStatus {
  initial,    // 초기 상태 
  ready,      // 분석 준비 상태 (추가된 상태)
  analyzing,  // 분석 중
  detailResult // 상세 결과 표시
}

// 분석 뷰모델 상태 클래스
class FinanceAnalysisState {
  final AnalysisStatus status;  // 현재 분석 상태
  final List<FinanceTypeModel> matchedTypes;  // 사용자에게 해당하는 유형들
  final FinanceTypeModel? selectedType;  // 최종 선택된 유형
  final bool isLoading;  // 로딩 중 여부
  final String? error;   // 에러 메시지

  // 생성자
  FinanceAnalysisState({
    this.status = AnalysisStatus.initial,
    this.matchedTypes = const [],
    this.selectedType,
    this.isLoading = false,
    this.error,
  });

  // 복사 생성자
  FinanceAnalysisState copyWith({
    AnalysisStatus? status,
    List<FinanceTypeModel>? matchedTypes,
    FinanceTypeModel? selectedType,
    bool? isLoading,
    String? error,
  }) {
    return FinanceAnalysisState(
      status: status ?? this.status,
      matchedTypes: matchedTypes ?? this.matchedTypes,
      selectedType: selectedType ?? this.selectedType,
      isLoading: isLoading ?? this.isLoading,
      error: error ?? this.error,
    );
  }
}

// 분석 뷰모델 클래스
class FinanceAnalysisViewModel extends StateNotifier<FinanceAnalysisState> {
  final Ref _ref;
  final Random _random = Random();

  // 생성자: 초기 상태 설정
  FinanceAnalysisViewModel(this._ref) : super(FinanceAnalysisState(status: AnalysisStatus.ready));

  // 자산 데이터 분석 시작 메서드
  Future<void> startAnalysis() async {
    try {
      // 로딩 상태로 변경
      state = state.copyWith(
        status: AnalysisStatus.analyzing,
        isLoading: true,
        error: null,
      );

      // 자산 데이터 확인 (이미 로드되어 있는지)
      final financeState = _ref.read(financeViewModelProvider);
      final assetData = financeState.assetData;
      
      if (assetData == null) {
        // 데이터가 없으면 자산 정보 가져오기
        await _ref.read(financeViewModelProvider.notifier).fetchAssetInfo();
        // 데이터 다시 확인
        final updatedState = _ref.read(financeViewModelProvider);
        if (updatedState.assetData == null) {
          throw Exception("자산 데이터를 불러올 수 없습니다.");
        }
      }

      // 유형 분석 실행
      final matchedTypes = _analyzeUserTypes();
      
      // 최소 6초 지연 추가 (여기에 추가)
      await Future.delayed(const Duration(seconds: 6));

      // 매칭된 유형이 없는 경우
      if (matchedTypes.isEmpty) {
        // 기본 유형을 하나 선택 (예: ID가 6인 유형)
        final defaultType = FinanceTypeConstants.getTypeById(5);
        state = state.copyWith(
          status: AnalysisStatus.detailResult,
          matchedTypes: [defaultType],
          selectedType: defaultType,
          isLoading: false,
        );
      } else {
        // 랜덤하게 하나의 유형 선택
        final selectedType = _selectRandomType(matchedTypes);
        state = state.copyWith(
          status: AnalysisStatus.detailResult,
          matchedTypes: matchedTypes,
          selectedType: selectedType,
          isLoading: false,
        );
      }
    } catch (e) {
      // 에러 처리
      state = state.copyWith(
        status: AnalysisStatus.initial,
        isLoading: false,
        error: e.toString(),
      );
    }
  }

  // 사용자 유형 분석 메서드
  List<FinanceTypeModel> _analyzeUserTypes() {
    final financeState = _ref.read(financeViewModelProvider);
    final assetData = financeState.assetData!.data;
    final List<FinanceTypeModel> matchedTypes = [];

    // 모든 유형 중에서 사용자에게 해당하는 유형 찾기
    final allTypes = FinanceTypeConstants.getAllTypes();
    
    for (var type in allTypes) {
      bool isMatched = false;
      
      // 유형별 조건 체크
      switch (type.id) {
        case 1: // 주택청약 꾸준히넣음
          isMatched = _checkHousingSubscription(assetData);
          break;
        case 2: // 소액대출
          isMatched = _checkSmallLoan(assetData);
          break;
        case 3: // 소액 적금 많음
          isMatched = _checkMultipleSmallSavings(assetData);
          break;
        case 4: // 카드 개수가 많음
          isMatched = _checkManyCards(assetData);
          break;
        case 5: // 적금 비율 높음
          isMatched = _checkHighSavingsRatio(assetData);
          break;
        case 6: // 총 예적금 금액 상위 10%
          isMatched = _checkHighTotalSavings(assetData);
          break;
      }
      
      if (isMatched) {
        matchedTypes.add(type);
      }
    }
    
    return matchedTypes;
    // return matchedTypes.isEmpty ? [] : [FinanceTypeConstants.getTypeById(6)];
  }

  // 랜덤하게 유형 선택 메서드
  // FinanceTypeModel _selectRandomType(List<FinanceTypeModel> types) {
  //   if (types.isEmpty) {
  //     throw Exception("매칭된 유형이 없습니다.");
  //   }
  //   // 무작위로 유형 하나 선택
  //   return types[_random.nextInt(types.length)];
  // }
FinanceTypeModel _selectRandomType(List<FinanceTypeModel> types) {
  if (types.isEmpty) {
    throw Exception("매칭된 유형이 없습니다.");
  }
  
  // 우선순위 순서 정의
  List<int> priorityOrder = [5, 4, 6, 3, 1, 2];
  
  // 우선순위 순서대로 타입 찾기
  for (int priority in priorityOrder) {
    for (var type in types) {
      if (type.id == priority) { // 여기서 id는 FinanceTypeModel의 식별자라고 가정합니다
        return type;
      }
    }
  }
  
  // 우선순위에 해당하는 타입이 없으면 첫 번째 타입 반환
  return types[0];
}

  // 분석 초기화 메서드
  void resetAnalysis() {
    state = FinanceAnalysisState();
  }

  // 유형 1: 주택청약 보유 체크
  bool _checkHousingSubscription(AssetData assetData) {
    // 예금, 적금, 대출 목록에서 "주택청약"을 포함하는 항목이 있는지 확인
    final hasDepositWithHousing = assetData.depositData.depositList.any(
      (deposit) => deposit.accountName.contains("주택청약")
    );
    
    final hasSavingsWithHousing = assetData.savingsData.savingsList.any(
      (savings) => savings.accountName.contains("주택청약")
    );
    
    final hasLoanWithHousing = assetData.loanData.loanList.any(
      (loan) => loan.accountName.contains("주택청약")
    );
    
    return hasDepositWithHousing || hasSavingsWithHousing || hasLoanWithHousing;
  }

  // 유형 2: 소액대출 체크
  bool _checkSmallLoan(AssetData assetData) {
    // 500만원 이하의 대출 보유 여부 확인
    return assetData.loanData.loanList.any(
      (loan) => (loan.loanBalance ?? 0) <= 5000000
    );
  }

  // 유형 3: 소액 적금 다수 보유 체크
  bool _checkMultipleSmallSavings(AssetData assetData) {
    // 100만원 이하 적금이 3개 이상인지 확인
    final smallSavings = assetData.savingsData.savingsList.where(
      (savings) => (savings.totalBalance ?? 0) <= 1000000
    ).toList();
    
    return smallSavings.length >= 3;
  }

  // 유형 4: 카드 다수 보유 체크
  bool _checkManyCards(AssetData assetData) {
    // 카드가 4개 이상인지 확인
    return assetData.cardData.cardList.length >= 4;
  }

  // 유형 5: 적금 비율 높음 체크
  bool _checkHighSavingsRatio(AssetData assetData) {
    // 적금 개수가 예금 개수보다 많은지 확인
    return assetData.savingsData.savingsList.length > assetData.depositData.depositList.length;
  }

  // 유형 6: 예적금 총액 높음 체크
  bool _checkHighTotalSavings(AssetData assetData) {
    // 예금 + 적금 총액이 1억 이상인지 확인
    final totalDeposit = int.tryParse(assetData.depositData.totalAmount) ?? 0;
    final totalSavings = int.tryParse(assetData.savingsData.totalAmount) ?? 0;
    
    return (totalDeposit + totalSavings) >= 100000000;
  }
}

// 분석 뷰모델 프로바이더 (StateNotifierProvider 사용)
final financeAnalysisViewModelProvider = StateNotifierProvider<FinanceAnalysisViewModel, FinanceAnalysisState>((ref) {
  return FinanceAnalysisViewModel(ref);
});