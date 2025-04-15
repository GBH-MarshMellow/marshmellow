// lib/presentation/viewmodels/finance/demand_detail_viewmodel.dart

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/finance_api.dart';
import 'package:marshmellow/data/models/finance/detail/demand_detail_model.dart';

// 입출금계좌 상세 뷰모델 프로바이더
final demandDetailViewModelProvider = Provider((ref) {
  return DemandDetailViewModel(ref);
});

// 입출금계좌 거래내역 상태 프로바이더
final demandTransactionsProvider = FutureProvider.family<DemandDetailResponse, DemandDetailParams>(
  (ref, params) async {
    final viewModel = ref.watch(demandDetailViewModelProvider);
    return await viewModel.getAccountTransactions(
      accountNo: params.accountNo,
      startDate: params.startDate,
      endDate: params.endDate,
      transactionType: params.transactionType,
      orderByType: params.orderByType,
    );
  },
);

// 뷰모델에 전달할 파라미터 클래스
class DemandDetailParams {
  final String accountNo;
  final String startDate;
  final String endDate;
  final String transactionType;
  final String? orderByType;

  DemandDetailParams({
    required this.accountNo,
    required this.startDate,
    required this.endDate,
    this.transactionType = 'A',
    this.orderByType,
  });
  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is DemandDetailParams &&
        other.accountNo == accountNo &&
        other.startDate == startDate &&
        other.endDate == endDate &&
        other.transactionType == transactionType &&
        other.orderByType == orderByType;
  }

  @override
  int get hashCode {
    return accountNo.hashCode ^
        startDate.hashCode ^
        endDate.hashCode ^
        transactionType.hashCode ^
        (orderByType?.hashCode ?? 0);
  }
}

// 필터 상태 프로바이더 (필요시 사용)
final transactionFilterProvider = StateProvider<String>((ref) => 'A');  // 기본값 'A' (전체)
final orderByTypeProvider = StateProvider<String?>((ref) => 'ASC');     // 기본값 'ASC'

class DemandDetailViewModel {
  final ProviderRef _ref;
  
  DemandDetailViewModel(this._ref);
  
  // API를 통해 계좌 거래내역 가져오기
  Future<DemandDetailResponse> getAccountTransactions({
    required String accountNo,
    required String startDate,
    required String endDate,
    String transactionType = 'A',
    String? orderByType,
  }) async {
    final financeApi = _ref.read(financeApiProvider);
    return await financeApi.getDemandAccountTransactions(
      accountNo: accountNo,
      startDate: startDate,
      endDate: endDate,
      transactionType: transactionType,
      orderByType: orderByType,
    );
  }
  
  // 필터 변경 메서드
  void changeTransactionFilter(String filter) {
    _ref.read(transactionFilterProvider.notifier).state = filter;
  }
  
  // 정렬 방식 변경 메서드
  void changeOrderType(String? orderType) {
    _ref.read(orderByTypeProvider.notifier).state = orderType;
  }
}