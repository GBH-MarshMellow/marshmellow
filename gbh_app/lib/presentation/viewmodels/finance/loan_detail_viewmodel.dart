// lib/presentation/viewmodels/finance/loan_detail_viewmodel.dart

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/finance_api.dart';
import 'package:marshmellow/data/models/finance/detail/loan_detail_model.dart';

// 대출계좌 상세 뷰모델 프로바이더
final loanDetailViewModelProvider = Provider((ref) {
  return LoanDetailViewModel(ref);
});

// 대출계좌 상세 정보 상태 프로바이더
final loanPaymentDetailsProvider = FutureProvider.family<LoanDetailResponse, String>(
  (ref, accountNo) async {
    final viewModel = ref.watch(loanDetailViewModelProvider);
    return await viewModel.getLoanPaymentDetails(accountNo: accountNo);
  },
);

class LoanDetailViewModel {
  final ProviderRef _ref;
  
  LoanDetailViewModel(this._ref);
  
  // API를 통해 대출계좌 상환 내역 가져오기
  Future<LoanDetailResponse> getLoanPaymentDetails({
    required String accountNo,
  }) async {
    final financeApi = _ref.read(financeApiProvider);
    return await financeApi.getLoanPaymentDetails(
      accountNo: accountNo,
    );
  }
}