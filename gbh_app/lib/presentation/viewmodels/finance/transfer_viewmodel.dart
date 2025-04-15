import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/finance_api.dart';
import 'package:marshmellow/data/models/finance/transfer_model.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'package:marshmellow/presentation/viewmodels/finance/demand_detail_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';

// 송금 단계 정의
enum TransferStep {
  accountInput,  // 계좌 정보 입력
  amountInput,   // 금액 입력
  loading,       // 송금 처리 중
  complete,      // 송금 완료
}

// 송금 상태 정의
class TransferState {
  final TransferStep step;
  final int withdrawalAccountId;  // 출금계좌 ID
  final String withdrawalAccountNo;  // 출금계좌 번호
  final String? selectedBankCode;  // 선택한 은행 코드
  final String? selectedBankName;  // 선택한 은행 이름
  final String depositAccountNo;  // 입금계좌 번호
  final int amount;  // 송금 금액
  final bool isLoading;
  final String? error;
  final bool isSuccess;  // 송금 성공 여부
  final String withdrawalBankName;  // 추가

  const TransferState({
    this.step = TransferStep.accountInput,
    this.withdrawalAccountId = 0,
    this.withdrawalAccountNo = '',
    this.selectedBankCode,
    this.selectedBankName,
    this.depositAccountNo = '',
    this.amount = 0,
    this.isLoading = false,
    this.error,
    this.isSuccess = false,
    this.withdrawalBankName = '',  // 추가
  });

  // 복사 생성자
  TransferState copyWith({
    TransferStep? step,
    int? withdrawalAccountId,
    String? withdrawalAccountNo,
    String? selectedBankCode,
    String? selectedBankName,
    String? depositAccountNo,
    int? amount,
    bool? isLoading,
    String? error,
    bool? isSuccess,
    String? withdrawalBankName,  // 추가
  }) {
    return TransferState(
      step: step ?? this.step,
      withdrawalAccountId: withdrawalAccountId ?? this.withdrawalAccountId,
      withdrawalAccountNo: withdrawalAccountNo ?? this.withdrawalAccountNo,
      selectedBankCode: selectedBankCode ?? this.selectedBankCode,
      selectedBankName: selectedBankName ?? this.selectedBankName,
      depositAccountNo: depositAccountNo ?? this.depositAccountNo,
      amount: amount ?? this.amount,
      isLoading: isLoading ?? this.isLoading,
      error: error,  // null 허용을 위해 ?? this.error 패턴 사용 안함
      isSuccess: isSuccess ?? this.isSuccess,
      withdrawalBankName: withdrawalBankName ?? this.withdrawalBankName,  // 추가
    );
  }

  // 계좌 정보가 모두 입력되었는지 확인
  bool get isAccountInputComplete => 
    selectedBankCode != null && 
    selectedBankName != null && 
    depositAccountNo.length == 16;

  // 송금 금액이 유효한지 확인 (0보다 커야함)
  bool get isAmountValid => amount > 0;
}

// 은행 목록 제공 (상수)
final List<Bank> bankList = [
  Bank(code: '001', name: '한국은행', iconPath: 'assets/icons/bank/001_korea.svg'),
  Bank(code: '002', name: '산업은행', iconPath: 'assets/icons/bank/002_kdb.svg'),
  Bank(code: '003', name: '기업은행', iconPath: 'assets/icons/bank/003_ibk.svg'),
  Bank(code: '004', name: '국민은행', iconPath: 'assets/icons/bank/004_kb.svg'),
  Bank(code: '011', name: '농협은행', iconPath: 'assets/icons/bank/011_nh.svg'),
  Bank(code: '020', name: '우리은행', iconPath: 'assets/icons/bank/020_woori.svg'),
  Bank(code: '023', name: 'SC은행', iconPath: 'assets/icons/bank/023_sc.svg'),
  Bank(code: '027', name: '씨티은행', iconPath: 'assets/icons/bank/027_citi.svg'),
  Bank(code: '032', name: '대구은행', iconPath: 'assets/icons/bank/032_daegu.svg'),
  Bank(code: '034', name: '광주은행', iconPath: 'assets/icons/bank/034_gwangju.svg'),
  Bank(code: '035', name: '제주은행', iconPath: 'assets/icons/bank/035_jeju.svg'),
  Bank(code: '037', name: '전북은행', iconPath: 'assets/icons/bank/037_junbuk.svg'),
  Bank(code: '039', name: '경남은행', iconPath: 'assets/icons/bank/039_gyeongnam.svg'),
  Bank(code: '045', name: 'MG새마을금고', iconPath: 'assets/icons/bank/045_mg.svg'),
  Bank(code: '081', name: '하나은행', iconPath: 'assets/icons/bank/081_hana.svg'),
  Bank(code: '088', name: '신한은행', iconPath: 'assets/icons/bank/088_shinhan.svg'),
  Bank(code: '090', name: '카카오뱅크', iconPath: 'assets/icons/bank/090_kakao.svg'),
  Bank(code: '999', name: '싸피뱅크', iconPath: 'assets/icons/bank/999_ssafy.svg'),
];

// 송금 ViewModel
class TransferViewModel extends StateNotifier<TransferState> {
  final Ref _ref;

  TransferViewModel(this._ref) : super(const TransferState());

  // 출금계좌 정보 설정
  void setWithdrawalAccount(int accountId, String accountNo) {
    state = state.copyWith(
      withdrawalAccountId: accountId,
      withdrawalAccountNo: accountNo,
    );
  }

  void setWithdrawalBankName(String bankName) {
    state = state.copyWith(withdrawalBankName: bankName);
  }

  // 은행 선택
  void selectBank(String code, String name) {
    state = state.copyWith(
      selectedBankCode: code,
      selectedBankName: name,
    );
  }

  // 입금계좌 번호 설정
  void setDepositAccountNo(String accountNo) {
    state = state.copyWith(depositAccountNo: accountNo);
  }

  // 다음 단계로 이동 (계좌 입력 → 금액 입력)
  void moveToAmountInput() {
    if (state.isAccountInputComplete) {
      state = state.copyWith(step: TransferStep.amountInput);
    }
  }

  // 송금 금액 설정
  void setAmount(int amount) {
    state = state.copyWith(amount: amount);
  }

  // 송금 실행 - bool 반환하도록 수정
  Future<bool> executeTransfer() async {
    if (!state.isAmountValid) return false;

    try {
      state = state.copyWith(
        isLoading: true, 
        error: null,
      );

      final request = TransferRequest(
        withdrawalAccountId: state.withdrawalAccountId,
        depositAccountNo: state.depositAccountNo,
        transactionSummary: '송금',  // 기본값
        transactionBalance: state.amount.toString(),
      );

      final financeApi = _ref.read(financeApiProvider);
      final response = await financeApi.transferMoney(request);

      // 송금 성공 시 자산 정보 새로고침
      if (response.code == 200) {
        final financeViewModel = _ref.read(financeViewModelProvider.notifier);
        await financeViewModel.refreshAssetInfo();
        _ref.invalidate(demandTransactionsProvider);

        state = state.copyWith(
          isLoading: false,
          isSuccess: true,
        );
        return true;
      } else {
        state = state.copyWith(
          isLoading: false,
          error: response.message,
        );
        return false;
      }
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: '송금 중 오류가 발생했습니다: ${e.toString()}',
      );
      return false;
    }
  }

  // 상태 초기화
  void reset() {
    state = const TransferState();
  }
}

// Provider 정의
final transferProvider = StateNotifierProvider<TransferViewModel, TransferState>((ref) {
  return TransferViewModel(ref);
});