class PinState {
  final String pin; // 사용자 입력 PIN 번호
  final bool isPinSet; // PIN이 이미 설정되어 있는지 여부
  final bool isBiometricsAvailable; // 기기에서 생체인식이 가능한지 여부
  final bool useBiometrics; // 사용자가 생체인식 사용을 설정했는지 여부
  final String errorMessage; // 오류 메시지(비어있으면 오류 없음)
  final bool isConfirmingPin; // PIN 확인 모드 여부
  final int currentDigit; // 현재 입력 중인 자릿수

  PinState({
    this.pin = '',
    this.isPinSet = false,
    this.isBiometricsAvailable = true,
    this.useBiometrics = false,
    this.errorMessage = '',
    this.isConfirmingPin = false,
    this.currentDigit = 0,
    
  });

  PinState copyWith({
    String? pin,
    bool? isPinSet,
    bool? isBiometricsAvailable,
    bool? useBiometrics,
    String? errorMessage,
    bool? isConfirmingPin,
    int? currentDigit,
  }) {
    return PinState(
      pin: pin ?? this.pin,
      isPinSet: isPinSet ?? this.isPinSet,
      isBiometricsAvailable: isBiometricsAvailable ?? this.isBiometricsAvailable,
      useBiometrics: useBiometrics ?? this.useBiometrics,
      errorMessage: errorMessage ?? this.errorMessage,
      isConfirmingPin: isConfirmingPin ?? this.isConfirmingPin,
      currentDigit: currentDigit ?? this.currentDigit,
    );
  }
}