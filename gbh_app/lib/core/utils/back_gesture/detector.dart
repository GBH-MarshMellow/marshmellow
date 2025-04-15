import 'package:flutter/cupertino.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/utils/back_gesture/controller.dart';

/*
  왼>오 스와이프 : 뒤로가기
*/
class SwipeBackDetector extends StatelessWidget {
  final Widget child;
  final BackGestureController controller;
  final GoRouter router;

  const SwipeBackDetector({
    Key? key,
    required this.child,
    required this.controller,
    required this.router,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // 컨트롤러의 상태를 구독
    return AnimatedBuilder(
      animation: controller, 
      builder: (context, _) {
        // 제스처가 비활성화된 경우 자식 위젯만 반환
        if (!controller.isGestureEnabled) {
          return child;
        }

        // 제스처가 활성화된 경우 GestureDector 포함
        return GestureDetector(
          onHorizontalDragEnd: (DragEndDetails details) {
            if (details.primaryVelocity != null && details.primaryVelocity! > 20) {
              if (router.canPop()) {
                router.pop();
              }
            }
          },
          behavior: HitTestBehavior.translucent,
          child: child,
        );
      }
    );
  }
}