// lib/presentation/viewmodels/finance/card_detail_viewmodel.dart

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/finance_api.dart';
import 'package:marshmellow/data/models/finance/detail/card_detail_model.dart';

// 카드 상세 뷰모델 프로바이더
final cardDetailViewModelProvider = Provider((ref) {
  return CardDetailViewModel(ref);
});

// 카드 거래내역 상태 프로바이더
final cardTransactionsProvider = FutureProvider.family<CardDetailResponse, CardDetailParams>(
  (ref, params) async {
    final viewModel = ref.watch(cardDetailViewModelProvider);
    return await viewModel.getCardTransactions(
      cardNo: params.cardNo,
      cvc: params.cvc,
      startDate: params.startDate,
      endDate: params.endDate,
    );
  },
);

// 뷰모델에 전달할 파라미터 클래스
class CardDetailParams {
  final String cardNo;
  final String cvc;
  final String startDate;
  final String endDate;

  CardDetailParams({
    required this.cardNo,
    required this.cvc,
    required this.startDate,
    required this.endDate,
  });
  
  // 무한 API 호출 방지를 위한 객체 비교 로직 추가
  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is CardDetailParams &&
        other.cardNo == cardNo &&
        other.cvc == cvc &&
        other.startDate == startDate &&
        other.endDate == endDate;
  }

  @override
  int get hashCode {
    return cardNo.hashCode ^
        cvc.hashCode ^
        startDate.hashCode ^
        endDate.hashCode;
  }
}

class CardDetailViewModel {
  final ProviderRef _ref;
  
  CardDetailViewModel(this._ref);
  
  // API를 통해 카드 거래내역 가져오기
  Future<CardDetailResponse> getCardTransactions({
    required String cardNo,
    required String cvc,
    required String startDate,
    required String endDate,
  }) async {
    final financeApi = _ref.read(financeApiProvider);
    return await financeApi.getCardTransactions(
      cardNo: cardNo,
      cvc: cvc,
      startDate: startDate,
      endDate: endDate,
    );
  }
}