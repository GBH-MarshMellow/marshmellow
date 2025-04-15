class NumberFormat {
  // 숫자에 천 단위 콤마 추가
  static String formatWithComma(String value) {
    if (value.isEmpty) return '0';
    
    // 숫자만 추출
    value = value.replaceAll(',', '');
    
    // 음수 부호 처리
    bool isNegative = value.startsWith('-');
    if (isNegative) {
      value = value.substring(1);
    }
    
    // 숫자로 변환
    int number = int.tryParse(value) ?? 0;
    
    // 천 단위 콤마 추가
    String formatted = number.toString().replaceAllMapped(
      RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
      (Match m) => '${m[1]},',
    );
    
    // 음수 부호 다시 추가
    return isNegative ? '-$formatted' : formatted;
  }
}