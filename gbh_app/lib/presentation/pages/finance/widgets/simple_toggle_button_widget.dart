// presentation/pages/finance/widgets/simple_toggle_button_widget.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';


class SimpleToggleButton extends ConsumerWidget {
  final bool isSimplePage;
  
  const SimpleToggleButton({
    Key? key, 
    this.isSimplePage = false,
  }) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isSimpleToggleOn = ref.watch(simpleViewModeProvider);
    
    return GestureDetector(
      onTap: () {
        // 토글 상태 변경
        ref.read(simpleViewModeProvider.notifier).state = !isSimpleToggleOn;
        
        // 페이지 이동 처리
        Future.delayed(const Duration(milliseconds: 300), () {
          if (isSimplePage) {
            // 간편 페이지에서는 메인 페이지로 이동
            context.replace(FinanceRoutes.root);
          } else {
            // 메인 페이지에서는 간편 페이지로 이동
            context.replace(FinanceRoutes.getSimplePath());
          }
        });
      },
child: AnimatedContainer(
  duration: const Duration(milliseconds: 300), // 애니메이션 시간 (0.3초)
  curve: Curves.easeInOut, // 부드러운 애니메이션 효과
  padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
  decoration: BoxDecoration(
    color: isSimpleToggleOn ? Colors.black : Colors.grey,
    borderRadius: BorderRadius.circular(6.0),
  ),
  child: const Text(
    "간편",
    style: TextStyle(
      color: Colors.white,
      fontWeight: FontWeight.bold,
    ),
  ),
)
    );
  }
}