import 'dart:math';
import 'package:flutter/material.dart';

/// 은행별 카드 색상 매핑 클래스
class BankColors {
  static final Map<String, Color> colors = {
    '농협은행': const Color(0xFF1E8754), // 녹색 계열 (NH 로고 색상) - 채도 낮춤
    '카카오뱅크': const Color(0xFFEFD566), // 노란색 (카카오 브랜드 컬러) - 채도 낮춤
    '한국은행': const Color(0xFF4975D6), // 파란색 (로고 및 홈페이지 CI) - 채도 낮춤
    '싸피은행': const Color(0xFF8CD0E8), // 채도 낮춤
    '신한은행': const Color(0xFF4975D6), // 파란색 (로고 및 홈페이지 CI) - 채도 낮춤
    '국민은행': const Color(0xFFE7CB4B), // 노란색 (KB 로고 색상) - 채도 낮춤
    '우리은행': const Color(0xFF5298C8), // 푸른색 (우리은행 공식 CI) - 채도 낮춤
    'KEB하나은행': const Color(0xFF3EA7A3), // 청록색 계열 (하나은행 CI) - 채도 낮춤
    '기업은행': const Color(0xFF435988), // 짙은 파란색 (IBK CI) - 채도 낮춤
    '토스뱅크': const Color(0xFF4D80D6), // 푸른색 (토스 브랜드 컬러) - 채도 낮춤
    '케이뱅크': const Color(0xFFD85E4B), // 빨간색 (k뱅크 브랜드 가이드) - 채도 낮춤
    '산업은행': const Color(0xFF4564A3), // 파란색 (KDB 공식 CI) - 채도 낮춤
    'SC제일은행': const Color(0xFF4DA58D), // 녹색 계열 (SC 로고 색상) - 채도 낮춤
    '시티은행': const Color(0xFF4A65A0), // 파란색 (Citi 로고 컬러) - 채도 낮춤
    '대구은행': const Color(0xFF4D91BC), // 푸른색 (DGB 대구은행 로고) - 채도 낮춤
    '광주은행': const Color(0xFF4D91C2), // 푸른색 (광주은행 CI) - 채도 낮춤
    '제주은행': const Color(0xFF4B80AC), // 푸른색 (제주은행 브랜드 가이드) - 채도 낮춤
    '전북은행': const Color(0xFF4D73A6), // 푸른색 (전북은행 CI) - 채도 낮춤
    '경남은행': const Color(0xFFB85757), // 빨간색 계열 (경남은행 CI) - 채도 낮춤
    '새마을금고': const Color(0xFF5298CC), // 푸른색 (MG 새마을금고 브랜드) - 채도 낮춤
  };
  
  // 랜덤 색상 생성을 위한 파스텔톤 색상 리스트
  static final List<Color> randomColors = [
    const Color(0xFFB5EAD7), // 파스텔 민트
    const Color(0xFFC7CEEA), // 파스텔 블루
    const Color(0xFFFFDAC1), // 파스텔 피치
    const Color(0xFFFFB7B2), // 파스텔 핑크
    const Color(0xFFE2F0CB), // 파스텔 라임
    const Color(0xFFD0E6FF), // 파스텔 스카이블루
    const Color(0xFFFFC6FF), // 파스텔 라벤더
    const Color(0xFFFFF1C6), // 파스텔 옐로우
    const Color(0xFFF0D5E5), // 파스텔 퍼플
    const Color(0xFFDDF2EB), // 파스텔 에메랄드
  ];
  
  /// 은행명으로 색상 가져오기
  /// 매핑된 색상이 없으면 랜덤 파스텔 색상 반환
  static Color getColorByBankName(String bankName) {
    // 매핑된 색상이 있으면 반환
    if (colors.containsKey(bankName)) {
      return colors[bankName]!;
    }
    
    // 없으면 랜덤 색상 반환
    final random = Random();
    return randomColors[random.nextInt(randomColors.length)];
  }
}