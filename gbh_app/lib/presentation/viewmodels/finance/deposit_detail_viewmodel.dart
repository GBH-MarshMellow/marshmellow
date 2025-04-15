// lib/presentation/viewmodels/finance/deposit_detail_viewmodel.dart

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/finance_api.dart';
import 'package:marshmellow/data/models/finance/detail/deposit_detail_model.dart';

// 예금계좌 상세 뷰모델 프로바이더
final depositDetailViewModelProvider = Provider((ref) {
  return DepositDetailViewModel(ref);
});

// 예금계좌 납입 정보 상태 프로바이더
final depositPaymentProvider = FutureProvider.family<DepositDetailResponse, String>(
  (ref, accountNo) async {
    final viewModel = ref.watch(depositDetailViewModelProvider);
    return await viewModel.getDepositPayment(accountNo: accountNo);
  },
);

class DepositDetailViewModel {
  final ProviderRef _ref;
  
  DepositDetailViewModel(this._ref);
  
  // API를 통해 예금 납입 정보 가져오기
  Future<DepositDetailResponse> getDepositPayment({
    required String accountNo,
  }) async {
    final financeApi = _ref.read(financeApiProvider);
    return await financeApi.getDepositPayment(
      accountNo: accountNo,
    );
  }
}