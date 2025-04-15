import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/models/ledger/payment_method.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/data/repositories/ledger/ledger_repository.dart';

// 결제수단 상태
class PaymentMethodState {
  final bool isLoading;
  final List<PaymentMethod> paymentMethods;
  final String? errorMessage;

  PaymentMethodState({
    this.isLoading = false,
    this.paymentMethods = const [],
    this.errorMessage,
  });

  PaymentMethodState copyWith({
    bool? isLoading,
    List<PaymentMethod>? paymentMethods,
    String? errorMessage,
  }) {
    return PaymentMethodState(
      isLoading: isLoading ?? this.isLoading,
      paymentMethods: paymentMethods ?? this.paymentMethods,
      errorMessage: errorMessage,
    );
  }
}

// 결제수단 ViewModel
class PaymentMethodViewModel extends StateNotifier<PaymentMethodState> {
  final LedgerRepository _repository;

  PaymentMethodViewModel(this._repository) : super(PaymentMethodState());

  // 결제수단 목록 로드
  Future<void> loadPaymentMethods() async {
    state = state.copyWith(isLoading: true, errorMessage: null);

    try {
      final paymentMethods = await _repository.getPaymentMethods();

      final cashMethod = PaymentMethod(
        bankCode: 'CASH',
        bankName: '현금',
        paymentType: 'CASH',
        paymentMethod: '현금',
      );

      state = state.copyWith(
        isLoading: false,
        paymentMethods: paymentMethods,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        errorMessage: e.toString(),
      );
    }
  }
}

// Provider 등록
final paymentMethodViewModelProvider =
    StateNotifierProvider<PaymentMethodViewModel, PaymentMethodState>((ref) {
  final repository = ref.watch(ledgerRepositoryProvider);
  return PaymentMethodViewModel(repository);
});
