  import 'package:flutter_riverpod/flutter_riverpod.dart';
  import 'package:marshmellow/data/datasources/remote/finance_api.dart';
  import 'package:marshmellow/data/models/finance/asset_response_model.dart';

  /// 자산 데이터 상태를 관리하는 클래스
  /// 데이터, 로딩 상태, 에러 정보를 포함
  class FinanceState {
  /// 자산 응답 데이터 (null일 수 있음 - 데이터 로드 전)
  final AssetResponseModel? assetData;
  
  /// 데이터 로딩 중 여부
  final bool isLoading;
  
  /// 에러 메시지 (에러 발생 시에만 값 존재)
  final String? error;

  /// 기본 생성자
  FinanceState({
    this.assetData,
    this.isLoading = false,
    this.error,
  });

  /// 불변성을 유지하면서 상태 일부만 업데이트하기 위한 복사 생성자
  /// 매개변수로 전달된 값만 새 값으로 대체하고 나머지는 기존 값 유지
  FinanceState copyWith({
    AssetResponseModel? assetData,
    bool? isLoading,
    String? error,
  }) {
    return FinanceState(
      assetData: assetData ?? this.assetData,
      isLoading: isLoading ?? this.isLoading,
      error: error ?? this.error,
    );
  }
  }

  /// 자산 데이터 관리를 위한 ViewModel (StateNotifier)
  /// 데이터 로드, 계산, 새로고침 등의 비즈니스 로직 포함
  class FinanceViewModel extends StateNotifier<FinanceState> {
  /// Riverpod의 Ref 객체 (다른 프로바이더 접근용)
  final Ref _ref;
  
  /// 생성자: 초기 상태는 빈 FinanceState
  /// @param ref Riverpod Ref 객체
  FinanceViewModel(this._ref) : super(FinanceState());
  
  /// 자산 정보를 가져오는 메소드
  /// 이미 데이터가 있으면 API 호출 생략 (캐싱 효과)
  Future<void> fetchAssetInfo() async {
    // 데이터가 이미 있으면 다시 로드하지 않음 (중복 API 호출 방지)
    if (state.assetData != null) return;
    
    try {
      // 로딩 상태로 변경
      state = state.copyWith(isLoading: true, error: null);
      
      // API 호출하여 데이터 가져오기
      final financeApi = _ref.read(financeApiProvider);
      // 테스트용 고정 userKey 사용 (실제 환경에서는 사용자 키 동적 처리 필요)
      final assetData = await financeApi.getAssetInfo();
      print('@@@@@기본@@@@@ : ${assetData}');
      // 가져온 데이터로 상태 업데이트
      state = state.copyWith(
        assetData: assetData,
        isLoading: false,
      );
    } catch (e) {
      print('@@@@@에러 1번 @@@@@ : ${e.toString()}');
      // 에러 발생 시 에러 메시지 저장
      state = state.copyWith(
        isLoading: false,
        error: e.toString(),
      );
    }
  }
  
  /// 총 자산 계산 메소드
  /// 각 자산 유형별 금액을 합산하고 부채를 차감
  /// @return 총 자산 금액 (데이터 없으면 0 반환)
  int calculateTotalAssets() {
    final assetData = state.assetData?.data;
    // 데이터가 없으면 0 반환
    if (assetData == null) return 0;
    
    // 각 자산 유형별 금액 합산 및 부채 차감
    return (int.tryParse(assetData.cardData.totalAmount) ?? 0) + 
          (int.tryParse(assetData.demandDepositData.totalAmount) ?? 0) + 
          (int.tryParse(assetData.savingsData.totalAmount) ?? 0) + 
          (int.tryParse(assetData.depositData.totalAmount) ?? 0) - 
          (int.tryParse(assetData.loanData.totalAmount) ?? 0);
  }
  
  /// 자산 데이터 강제 새로고침 메소드
  /// 기존 데이터 유무와 관계없이 API를 호출하여 최신 데이터로 갱신
  Future<void> refreshAssetInfo() async {
    try {
      // 로딩 상태로 변경
      state = state.copyWith(isLoading: true, error: null);
      
      // API 호출하여 데이터 새로 가져오기
      final financeApi = _ref.read(financeApiProvider);
      final assetData = await financeApi.getAssetInfo();
      print('@@@@@리셋@@@@@ : ${assetData}');
      
      // 새로 가져온 데이터로 상태 업데이트
      state = state.copyWith(
        assetData: assetData,
        isLoading: false,
      );
    } catch (e) {
      print('@@@@@에러 2번 @@@@@ :${e.toString()}');
      // 에러 발생 시 에러 메시지 저장
      state = state.copyWith(
        isLoading: false,
        error: e.toString(),
      );
    }
  }
  }

  /// 자산 뷰모델 프로바이더 (StateNotifierProvider 사용)
  /// 앱 전체에서 동일한 자산 데이터 상태 공유 가능
  final financeViewModelProvider = StateNotifierProvider<FinanceViewModel, FinanceState>((ref) {
  return FinanceViewModel(ref);
  });

  /// 간편 모드 토글 상태를 위한 Provider
  /// 자산 정보 표시 방식 (상세/간편) 제어
  final simpleViewModeProvider = StateProvider<bool>((ref) => false);

  /// 자산 숨김버튼 상태 프로바이더
  /// 자산 정보 표시/숨김 제어
  final isFinanceHideProvider = StateProvider<bool>((ref) => false);