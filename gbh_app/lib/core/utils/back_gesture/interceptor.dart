import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';

/*
  왼>오 스와이프 : 뒤로가기
  뒤로가기 버튼 제어 위젯
*/
class BackButtonInterceptor extends StatelessWidget {
  final Widget child;
  final bool allowBackButton;
  final VoidCallback? onBackPressed;

  const BackButtonInterceptor({
    Key? key,
    required this.child,
    this.allowBackButton = true,
    this.onBackPressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return PopScope(
      // 사용자가 물리적 뒤로가기 버튼을 누르거나 시스템 뒤로가기 제스처를 사용할 때
      // pop 동작 허용 여부
      canPop: allowBackButton,

      // onPopInvoked : pop 동작이 호출될 때 실행될 콜백
      // canPop false 여도 호출됩니다
      onPopInvoked: (didPop) {
        if (didPop) {
          if (onBackPressed != null) {
            onBackPressed!();
          }
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('이 화면에서는 뒤로가기를 사용할 수 없습니다'),
              duration: Duration(seconds: 2),
            ),
          );
        }
      },
      child: child,
    );
  }
}