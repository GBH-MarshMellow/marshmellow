// TopTriangleBubbleWidget.dart
import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class TopTriangleBubbleWidget extends StatefulWidget {
  final String text;
  final Color baseBackgroundColor;
  final Color pulseBackgroundColor; 
  final Color textColor;
  final double width;
  final double height;
  final EdgeInsetsGeometry margin;
  final TextStyle? textStyle;
  final Duration pulseDuration;

  const TopTriangleBubbleWidget({
    Key? key,
    required this.text,
    this.baseBackgroundColor = const Color.fromARGB(255, 211, 211, 211),
    this.pulseBackgroundColor = const Color.fromARGB(255, 180, 180, 180), // 깜빡일 때의 색상
    this.textColor = Colors.white,
    this.width = 45,
    this.height = 30,
    this.margin = const EdgeInsets.only(right: 40),
    this.textStyle,
    this.pulseDuration = const Duration(seconds: 2), // 깜빡임 한 사이클의 기간
  }) : super(key: key);

  @override
  State<TopTriangleBubbleWidget> createState() => _TopTriangleBubbleWidgetState();
}

class _TopTriangleBubbleWidgetState extends State<TopTriangleBubbleWidget> with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<Color?> _colorAnimation;

  @override
  void initState() {
    super.initState();
    
    _animationController = AnimationController(
      vsync: this,
      duration: widget.pulseDuration,
    );
    
    _colorAnimation = ColorTween(
      begin: widget.baseBackgroundColor,
      end: widget.pulseBackgroundColor,
    ).animate(
      CurvedAnimation(
        parent: _animationController,
        curve: Curves.easeInOut,
      ),
    );
    
    // 애니메이션이 앞뒤로 반복되도록 설정
    _animationController.repeat(reverse: true);
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animationController,
      builder: (context, child) {
        return Container(
          margin: widget.margin,
          child: CustomPaint(
            painter: TopTriangleBubblePainter(
              color: _colorAnimation.value ?? widget.baseBackgroundColor,
            ),
            child: Container(
              width: widget.width,
              height: widget.height,
              padding: const EdgeInsets.fromLTRB(0, 6, 0, 0),
              alignment: Alignment.center,
              child: Text(
                widget.text,
                style: widget.textStyle ?? AppTextStyles.bodySmall.copyWith(
                  color: widget.textColor,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

class TopTriangleBubblePainter extends CustomPainter {
  final Color color;
  
  TopTriangleBubblePainter({required this.color});
  
  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    
    final Path path = Path();
    
    // 말풍선의 본체 부분 (둥근 직사각형)
    final RRect bubble = RRect.fromRectAndRadius(
      Rect.fromLTWH(0, 6, size.width, size.height - 6), // 위쪽에 삼각형 공간 확보
      Radius.circular(8),
    );
    
    // 상단 중앙에 삼각형 추가
    final triangleWidth = 12.0;
    final triangleHeight = 6.0;
    final triangleStartX = (size.width - triangleWidth) / 2;
    
    path.moveTo(triangleStartX, 6); // 삼각형 왼쪽 아래
    path.lineTo(triangleStartX + triangleWidth/2, 0); // 삼각형 꼭대기
    path.lineTo(triangleStartX + triangleWidth, 6); // 삼각형 오른쪽 아래
    path.close();
    
    // 본체와 삼각형 그리기
    canvas.drawRRect(bubble, paint);
    canvas.drawPath(path, paint);
  }
  
  @override
  bool shouldRepaint(covariant TopTriangleBubblePainter oldDelegate) {
    return oldDelegate.color != color;
  }
}