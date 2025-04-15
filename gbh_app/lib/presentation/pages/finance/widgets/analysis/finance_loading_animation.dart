import 'dart:math';
import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';

// 분석 화면 상태 열거형 (위젯 내부용)
enum LoadingAnimationState {
  ready,      // 준비 상태 (정지된 애니메이션)
  analyzing,  // 분석 중 (동작하는 애니메이션)
}

// 준비 상태와 애니메이션 상태를 모두 포함하는 애니메이션 위젯
class FinanceLoadingAnimation extends StatefulWidget {
  final LoadingAnimationState state;  // 현재 상태
  final VoidCallback onStartPressed;  // 시작 버튼 클릭 핸들러
  final int? resultTypeId;            // 결과 유형 ID (1~6)

  const FinanceLoadingAnimation({
    Key? key,
    required this.state,
    required this.onStartPressed,
    this.resultTypeId,
  }) : super(key: key);

  @override
  State<FinanceLoadingAnimation> createState() => _FinanceLoadingAnimationState();
}

class _FinanceLoadingAnimationState extends State<FinanceLoadingAnimation> 
    with SingleTickerProviderStateMixin {
  // 애니메이션 컨트롤러
  late AnimationController _controller;
  late Animation<double> _animation;
  
  // 회전 각도 계산 (라디안)
  double get _finalAngle {
    // null일 경우 4번 유형(120도)로 처리
    final tempTypeNum = 5;
    // 뷰모델 코드 final defaultType = FinanceTypeConstants.getTypeById(5);
    final typeId = widget.resultTypeId ?? tempTypeNum;
    
    // 유형 ID에 따른 각도 매핑 (회전 방향 고려)
    final Map<int, double> typeToAngle = {
      1: 0.0,
      2: 30.0,
      3: 60.0,
      4: 90.0,
      5: 120.0,
      6: 150.0,
    };
    
    // 각도 가져오기 (기본값 120도)
    final degreeAngle = typeToAngle[typeId] ?? tempTypeNum*30.0;
    
    return -1 * degreeAngle * (pi / 180); // 라디안 변환
  }

  @override
  void initState() {
    super.initState();
    // 애니메이션 컨트롤러 초기화 (분석 애니메이션은 5초 동안 진행)
    // viewmodel에서 총시간은 6초로 해둠 (await Future.delayed(const Duration(seconds: 6));)
    // 즉 1초 쉬었다가 다음화면 넘어감
    _controller = AnimationController(
      duration: const Duration(seconds: 5),
      vsync: this,
    );
    
    // 처음엔 빠르게 회전하다가 천천히 멈추는 애니메이션
    _animation = CurvedAnimation(
      parent: _controller,
      // 처음에 빠르게 가속하다 끝에 천천히 감속하는 커브
      curve: Curves.easeInOut,
    );
    
    // 상태에 따른 애니메이션 제어
    _updateAnimationState();
  }

  @override
  void didUpdateWidget(FinanceLoadingAnimation oldWidget) {
    super.didUpdateWidget(oldWidget);
    // 상태가 변경되면 애니메이션 제어
    if (widget.state != oldWidget.state) {
      _updateAnimationState();
    }
  }
  
  // 상태에 따른 애니메이션 제어 메소드
  void _updateAnimationState() {
    if (widget.state == LoadingAnimationState.analyzing) {
      // 분석 중일 때는 애니메이션 시작
      _controller.reset();
      _controller.forward();
    } else {
      // 준비 상태일 때는 애니메이션 정지
      _controller.stop();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 화면 크기 계산
    final screenHeight = MediaQuery.of(context).size.height;
    final screenWidth = MediaQuery.of(context).size.width;
    
    // 룰렛 크기 및 위치 계산
    final rouletteSize = screenWidth * 2.3; // 화면 너비보다 크게
    final bottomPadding = screenHeight * 0.17; // 룰렛 중심이 화면 밖 아래쪽에 위치하게 함
    
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Text(
          '나의 유형을 테스트 해보세요',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w600,
            letterSpacing: -1.6,
            color: AppColors.greenPrimary,),
          textAlign: TextAlign.center,
        ),
        // const SizedBox(height: 16),
        const Text(
          '자산 유형 분석',
          style: TextStyle(
            fontSize: 40,
            fontWeight: FontWeight.w600,
            letterSpacing: -1.6,
            color: AppColors.textPrimary,),
        ),
        // const SizedBox(height: 16),
        // const Text(
        //   '당신의 자산 데이터를 기반으로\n맞춤형 자산 분석을 제공합니다',
        //   style: TextStyle(
        //     fontSize: 16,
        //     fontWeight: FontWeight.w400,
        //     letterSpacing: -0.6,
        //     color: AppColors.greenPrimary,),
        //   textAlign: TextAlign.center,
        // ),
        // const SizedBox(height: 32),
        
        // 룰렛 애니메이션
        ClipRRect(
          borderRadius: BorderRadius.all(Radius.circular(20.0)),
          child: Container(
            height: screenHeight * 0.43, // 화면 높이의 30% 정도 차지
            width: double.infinity,
            // color: Colors.amber,
            // decoration: BoxDecoration(
            //   border: Border.all(width: 4.0),
            //   borderRadius: BorderRadius.all(Radius.circular(20.0))
            // ),
            child: AnimatedBuilder(
              animation: _animation,
              builder: (context, child) {
                return Stack(
                  alignment: Alignment.center,
                  children: [
                    // 룰렛 이미지 (회전)
                    Positioned(
                      bottom: -rouletteSize / 2 -bottomPadding, // 중심이 화면 밖 아래쪽에 위치
                      child: Transform.rotate(
                        // 애니메이션 진행 중이면 부드럽게 회전, 아니면 결과 각도로 고정
                        angle: widget.state == LoadingAnimationState.analyzing
                            ? (_animation.value * -10 * pi) + (_finalAngle * _animation.value) // 여러 바퀴 회전 후 최종 각도에 도달
                            : 0, // 초기 상태는 0도
                        child: Image.asset(
                          'assets/images/finance/finance_analysis_circle.png',
                          width: rouletteSize,
                          height: rouletteSize,
                        ),
                      ),
                    ),
                    
                    // // 화살표 표시 (고정)
                    // Positioned(
                    //   bottom: bottomPadding,
                    //   child: const Icon(
                    //     Icons.arrow_drop_down,
                    //     size: 48,
                    //     color: Colors.red,
                    //   ),
                    // ),
                  ],
                );
              },
            ),
          ),
        ),
        
        const SizedBox(height: 24),
        
        // 상태에 따른 텍스트 표시
        if (widget.state == LoadingAnimationState.analyzing)
          Button(
            text: '분석 중...',
            width: MediaQuery.of(context).size.width * 0.35,
            onPressed: () {
            },
          ),

          
        
        // 상태에 따라 버튼 표시 여부 결정
        if (widget.state == LoadingAnimationState.ready)
          Button(
            text: '분석시작',
            width: MediaQuery.of(context).size.width * 0.35,
            onPressed: widget.onStartPressed,
          ),
      ],
    );
  }
}