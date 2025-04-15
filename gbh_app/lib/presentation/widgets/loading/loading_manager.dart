// lib/presentation/widgets/loading/loading_manager.dart
import 'package:flutter/material.dart';
import 'custom_loading_indicator.dart';

class LoadingManager {
  static OverlayEntry? _overlayEntry;
  static bool _isVisible = false;
  static DateTime? _showTime; // 로딩 표시 시작 시간
  static Duration _minimumDuration = const Duration(seconds: 1); // 최소 표시 시간

  // 로딩 인디케이터 표시
  static void show(
    BuildContext context, {
    String text = "이곳에 text 입력",
    double opacity = 0.7,
    Color backgroundColor = Colors.black, // 배경색 파라미터 추가
    int durationInSeconds = 1,
    int minimumDurationInSeconds = 1, // 최소 표시 시간 매개변수 추가
  }) {
    if (_isVisible) {
      // 이미 표시 중이면 제거
      hide();
    }

    // 표시 시작 시간 기록
    _showTime = DateTime.now();
    
    // 최소 표시 시간 설정
    _minimumDuration = Duration(seconds: minimumDurationInSeconds);


    // OverlayEntry 생성
    _overlayEntry = OverlayEntry(
      builder: (context) => CustomLoadingIndicator(
        text: text,
        opacity: opacity,
        backgroundColor: backgroundColor,
      ),
    );

    // Overlay에 추가
    Overlay.of(context).insert(_overlayEntry!);
    _isVisible = true;

    // 지정된 시간 후 자동으로 닫기
    if (durationInSeconds > 0) {
      Future.delayed(Duration(seconds: durationInSeconds), () {
        hide();
      });
    }
  }

  // 로딩 인디케이터 숨기기 (최소 표시 시간 적용)
  static Future<void> hide() async {
    if (!_isVisible || _overlayEntry == null) {
      return;
    }
    
    // 현재 시간과 표시 시작 시간의 차이 계산
    final elapsedTime = DateTime.now().difference(_showTime!);
    
    // 최소 표시 시간보다 적게 지났다면 차이만큼 대기
    if (elapsedTime < _minimumDuration) {
      await Future.delayed(_minimumDuration - elapsedTime);
    }
    
    // 로딩 인디케이터 제거
    if (_overlayEntry != null && _isVisible) {
      _overlayEntry!.remove();
      _overlayEntry = null;
      _isVisible = false;
      _showTime = null;
    }
  }
}