// 상태 정의
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/models/my/salary_model.dart';
import 'package:marshmellow/data/repositories/my/salary_repository.dart';
import 'package:marshmellow/di/providers/my/salary_provider.dart';

class SalaryState {
  final bool isLoading;
  final String? errorMessage;
  final List<AccountModel> accounts;
  final List<DepositModel> deposits;
  final AccountModel? selectedAccount;
  final DepositModel? selectedDeposit;
  final SalaryModel? salary;

  SalaryState({
    this.isLoading = false,
    this.errorMessage,
    this.accounts = const [],
    this.deposits = const [],
    this.selectedAccount,
    this.selectedDeposit,
    this.salary,
  });

  SalaryState copyWith({
    bool? isLoading,
    String? errorMessage,
    List<AccountModel>? accounts,
    List<DepositModel>? deposits,
    AccountModel? selectedAccount,
    DepositModel? selectedDeposit,
    SalaryModel? salary,
    bool clearError = false,
  }) {
    return SalaryState(
      isLoading: isLoading ?? this.isLoading,
      errorMessage: clearError ? null : (errorMessage ?? this.errorMessage),
      accounts: accounts ?? this.accounts,
      deposits: deposits ?? this.deposits,
      selectedAccount: selectedAccount ?? this.selectedAccount,
      selectedDeposit: selectedDeposit ?? this.selectedDeposit,
      salary: salary ?? this.salary,
    );
  }
}

// SalaryNotifier 구현
class SalaryNotifier extends StateNotifier<SalaryState> {
  final MySalaryRepository _repository;

  SalaryNotifier(this._repository) : super(SalaryState());

  // 입출금계좌목록 조회
  Future<void> fetchAccounts() async {
    try {
      state = state.copyWith(isLoading: true, clearError: true);
      final accounts = await _repository.getAccountList();
      state = state.copyWith(
        isLoading: false,
        accounts: accounts,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
    }
  }

  // 계좌 선택
  void selectAccount(AccountModel account) {
    state = state.copyWith(
      selectedAccount: account,
      deposits: [], // 계좌가 변경되면 기존 입금 내역 초기화
      selectedDeposit: null,
    );
  }

  // 선택한 계좌의 입금 내역 조회
  Future<void> fetchDeposits() async {
    if (state.selectedAccount == null) {
      state = state.copyWith(
        errorMessage: '선택된 계좌가 없습니다.',
      );
      return;
    }

    try {
      state = state.copyWith(isLoading: true, clearError: true);
      final deposits = await _repository.getDepositList(state.selectedAccount!.accountNo);
      state = state.copyWith(
        isLoading: false,
        deposits: deposits,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
    }
  }

  // 입금 내역 선택
  void selectDeposit(DepositModel deposit) {
    state = state.copyWith(selectedDeposit: deposit);
  }

  // 선택된 내역 확인 메서드 추가
  bool isDepositSelected(DepositModel deposit) {
    if (state.selectedDeposit == null) return false;
    
    // 날짜, 시간, 금액으로 비교
    return state.selectedDeposit!.transactionDate == deposit.transactionDate &&
          state.selectedDeposit!.transactionTime == deposit.transactionTime &&
          state.selectedDeposit!.transactionBalance == deposit.transactionBalance;
  }

  // 월급 등록 (선택한 입금 내역 기준)
  Future<bool> registerSalaryFromDeposit() async {
    if (state.selectedDeposit == null) {
      state = state.copyWith(
        errorMessage: '선택된 입금 내역이 없습니다.',
      );
      return false;
    }

    try {
      state = state.copyWith(isLoading: true, clearError: true);
      
      // 날짜에서 일자만 추출 (예: "20250328" -> 28)
      final dateStr = state.selectedDeposit!.transactionDate;
      final day = int.parse(dateStr.substring(6, 8));
      
      final success = await _repository.registerSalary(
        state.selectedDeposit!.transactionBalance,
        day,
      );
      
      if (success) {
        // 등록 성공 시 월급 정보 업데이트
        final salary = SalaryModel(
          salary: state.selectedDeposit!.transactionBalance,
          date: day,
        );
        
        state = state.copyWith(
          isLoading: false,
          salary: salary,
        );
      }
      
      return success;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
      return false;
    }
  }

  // 월급 정보 조회
  // Future<void> fetchSalaryInfo() async {
  //   try {
  //     state = state.copyWith(isLoading: true, clearError: true);
  //     final salary = await _repository.getSalaryInfo();
  //     state = state.copyWith(
  //       isLoading: false,
  //       salary: salary,
  //     );
  //   } catch (e) {
  //     state = state.copyWith(
  //       isLoading: false,
  //       errorMessage: e.toString(),
  //     );
  //   }
  // }

  // 월급 정보 수동 업데이트
  Future<bool> updateSalary(int salary, int date) async {
    try {
      state = state.copyWith(isLoading: true, clearError: true);
      final success = await _repository.updateSalary(salary, date);
      
      if (success) {
        // 수정 성공 시 월급 정보 업데이트
        final salaryModel = SalaryModel(
          salary: salary,
          date: date,
        );
        
        state = state.copyWith(
          isLoading: false,
          salary: salaryModel,
        );
      }
      
      return success;
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
      return false;
    }
  }
}

// Provider 정의
final mySalaryProvider = StateNotifierProvider<SalaryNotifier, SalaryState>((ref) {
  final repository = ref.watch(mySalaryRepositoryProvider);
  return SalaryNotifier(repository);
});