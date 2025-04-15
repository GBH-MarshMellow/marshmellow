import 'package:flutter_riverpod/flutter_riverpod.dart';

// 모달 상태를 위한 StateNotifier
class ModalNotifier extends StateNotifier<bool> {
  ModalNotifier() : super(false);

  void showModal() => state = true;
  void hideModal() => state = false;
}

// 모달 상태 Provider
final modalProvider = StateNotifierProvider<ModalNotifier, bool>((ref) {
  return ModalNotifier();
});
