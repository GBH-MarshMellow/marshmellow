// lib/presentation/viewmodels/finance/saving_detail_viewmodel.dart

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/finance_api.dart';
import 'package:marshmellow/data/models/finance/detail/saving_detail_model.dart';

// 적금계좌 상세 뷰모델 프로바이더
final savingDetailViewModelProvider = Provider((ref) {
  return SavingDetailViewModel(ref);
});

// 적금계좌 납입내역 상태 프로바이더
// 파라미터 객체가 아닌 계좌번호만 받도록 단순화
final savingPaymentsProvider = FutureProvider.family<SavingDetailResponse, String>(
  (ref, accountNo) async {
    final viewModel = ref.watch(savingDetailViewModelProvider);
    return await viewModel.getSavingPayments(accountNo: accountNo);
  },
);

class SavingDetailViewModel {
  final ProviderRef _ref;
  
  SavingDetailViewModel(this._ref);
  
  // API를 통해 적금 납입내역 가져오기
  Future<SavingDetailResponse> getSavingPayments({
    required String accountNo,
  }) async {
    final financeApi = _ref.read(financeApiProvider);
    return await financeApi.getSavingAccountPayments(
      accountNo: accountNo,
    );
  }
}