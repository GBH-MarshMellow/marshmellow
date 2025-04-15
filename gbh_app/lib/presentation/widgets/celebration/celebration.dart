import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

/// 커스터마이즈 가능한 축하 위젯
/// 텍스트, 이미지, 지속 시간 등을 설정 가능
class CelebrationPopup extends StatefulWidget {
  /// 메인 타이틀 텍스트
  final String titleText;

  /// 서브 텍스트
  final String subtitleText;

  /// 캐릭터 이미지 경로
  final String characterImagePath;

  /// 컨페티 지속 시간 (밀리초)
  final int confettiDuration;

  /// 컨페티 개수
  final int confettiCount;

  /// 배경 투명도 (0.0 ~ 1.0)
  final double backgroundOpacity;

  /// 위젯 닫기 콜백
  final VoidCallback? onClose;

  /// 유한슬 커스텀
  final bool showTexts;
  final bool showConfetti;

  const CelebrationPopup({
    Key? key,
    this.titleText = '야호!',
    this.subtitleText = '퇴사 질러~~~',
    this.characterImagePath = 'assets/images/characters/char_jump.png',
    this.confettiDuration = 4000,
    this.confettiCount = 10, 
    this.showTexts = true,
    this.showConfetti = true,
    this.backgroundOpacity = 0.5, // 기본 투명도
    this.onClose,
  }) : super(key: key);

  @override
  State<CelebrationPopup> createState() => _CelebrationPopupState();
}

class _CelebrationPopupState extends State<CelebrationPopup>
    with TickerProviderStateMixin {
  // 애니메이션 컨트롤러
  late AnimationController _characterController;
  late AnimationController _confettiController;
  late Animation<double> _characterAnimation;

  // 컨페티 리스트
  final List<SimpleConfettiItem> _confettiItems = [];
  final Random _random = Random();

  @override
  void initState() {
    super.initState();

    // 캐릭터 애니메이션 - 위아래로 약간 통통 뛰는 듯한 효과
    _characterController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _characterAnimation = Tween<double>(begin: -10.0, end: 10.0).animate(
      CurvedAnimation(parent: _characterController, curve: Curves.easeInOut),
    );

    // 컨페티 애니메이션
    _confettiController = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: widget.confettiDuration),
    )..forward();

    // 컨페티 아이템 생성
    _generateConfetti();

    // 컨페티 애니메이션 리스너
    _confettiController.addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        // 컨페티 애니메이션이 끝나면 위젯을 자동으로 닫을 수도 있음
        // widget.onClose?.call();
      }
    });
  }

  void _generateConfetti() {
    // 사용할 컨페티 이미지 경로 리스트
    final List<String> confettiPaths = [
      IconPath.confettiGreen,
      IconPath.confettiPink,
      IconPath.confettiYellow1,
      IconPath.confettiYellow2,
      IconPath.confettiBlue,
    ];

    // 폭발 중심점
    final double centerX = 140.0;
    final double centerY = 230.0;

    // 다양한 방향으로 컨페티 생성
    for (int i = 0; i < widget.confettiCount; i++) {
      // 폭발 각도 (360도 방향 랜덤)
      final double angle = _random.nextDouble() * 2 * pi;

      // 폭발 거리 (화면 전체에 고르게 분포)
      final double distance = _random.nextDouble() * 300 + 50;

      // 목표 위치 계산
      final double targetX = centerX + cos(angle) * distance;
      final double targetY = centerY + sin(angle) * distance;

      // 속도 (빠르게 이동)
      final double speed = _random.nextDouble() * 0.4 + 1;

      // 크기 랜덤화
      final double size = (_random.nextDouble() * 30 + 20) * 3.5;

      // 회전 효과
      final double rotation = _random.nextDouble() * 2 * pi;
      final double rotationSpeed = (_random.nextDouble() - 0.5) * 0.8;

      // 지연 시간 (거의 동시에 폭발)
      final double delay = _random.nextDouble() * 0.2;

      // 랜덤 컨페티 이미지 선택
      final String path = confettiPaths[_random.nextInt(confettiPaths.length)];

      // 컨페티 아이템 생성 및 추가
      _confettiItems.add(SimpleConfettiItem(
        path: path,
        centerX: centerX,
        centerY: centerY,
        targetX: targetX,
        targetY: targetY,
        size: size,
        rotation: rotation,
        rotationSpeed: rotationSpeed,
        speed: speed,
        delay: delay,
      ));
    }
  }

  @override
  void dispose() {
    _characterController.dispose();
    _confettiController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: GestureDetector(
        onTap: widget.onClose,
        child: Container(
          width: double.infinity,
          height: double.infinity,
          color: AppColors.whiteDark
              .withOpacity(widget.backgroundOpacity), // 투명도 조절 가능
          child: Stack(
            children: [
              // 각 컨페티 아이템 렌더링
              ..._confettiItems.map((item) => AnimatedBuilder(
                    animation: _confettiController,
                    builder: (context, child) {
                      final SimpleConfettiItem confettiItem = item;

                      // 애니메이션 진행 시간 계산 (지연 시간 고려)
                      final double time =
                          (_confettiController.value - confettiItem.delay)
                              .clamp(0.0, 1.0);

                      // 컨페티가 지연 시간 전이면 렌더링하지 않음
                      if (time <= 0) return const SizedBox.shrink();

                      // 현재 위치 계산 (직선 이동)
                      final currentX = confettiItem.centerX +
                          (confettiItem.targetX - confettiItem.centerX) *
                              (time * confettiItem.speed);

                      final currentY = confettiItem.centerY +
                          (confettiItem.targetY - confettiItem.centerY) *
                              (time * confettiItem.speed);

                      // 크기 및 투명도 애니메이션
                      // 빠르게 나타났다가 천천히 사라짐
                      double scale = 1.0;
                      double opacity = 1.0;

                      if (time < 0.2) {
                        // 빠르게 나타남 (0->1)
                        scale = time / 0.2;
                        opacity = time / 0.2;
                      } else if (time > 0.8) {
                        // 천천히 사라짐 (1->0)
                        opacity = 1.0 - ((time - 0.8) / 0.2);
                      }

                      // 현재 회전 계산
                      final currentRotation = confettiItem.rotation +
                          time * 10 * confettiItem.rotationSpeed;

                      return Positioned(
                        left: currentX,
                        top: currentY,
                        child: Opacity(
                          opacity: opacity,
                          child: Transform.rotate(
                            angle: currentRotation,
                            child: Transform.scale(
                              scale: scale,
                              child: Image.asset(
                                confettiItem.path,
                                width: confettiItem.size,
                                height: confettiItem.size,
                              ),
                            ),
                          ),
                        ),
                      );
                    },
                  )),

              // 중앙 컨텐츠 (캐릭터와 텍스트)
              Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // 캐릭터 (위아래로 약간 움직임)
                    AnimatedBuilder(
                      animation: _characterAnimation,
                      builder: (context, child) {
                        return Transform.translate(
                          offset: Offset(0, _characterAnimation.value),
                          child: Image.asset(
                            widget.characterImagePath,
                            width: 200,
                            height: 200,
                          ),
                        );
                      },
                    ),
                    const SizedBox(height: 20),
                    // 메인 타이틀 텍스트
                    Text(
                      widget.titleText,
                      style: AppTextStyles.bodyExtraLarge.copyWith(
                        fontWeight: FontWeight.w600,
                        color: AppColors.textPrimary,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    // 서브 텍스트
                    Text(
                      widget.subtitleText,
                      style: AppTextStyles.bodyExtraLarge.copyWith(
                        fontWeight: FontWeight.w600,
                        color: AppColors.textPrimary,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// 컨페티 아이템 클래스 (단순화된 버전)
class SimpleConfettiItem {
  final String path;
  final double centerX;
  final double centerY;
  final double targetX;
  final double targetY;
  final double size;
  final double rotation;
  final double rotationSpeed;
  final double speed;
  final double delay;

  SimpleConfettiItem({
    required this.path,
    required this.centerX,
    required this.centerY,
    required this.targetX,
    required this.targetY,
    required this.size,
    required this.rotation,
    required this.rotationSpeed,
    required this.speed,
    required this.delay,
  });
}

/// 축하 팝업을 보여주는 간편 함수
/// 텍스트와 이미지를 커스터마이즈 할 수 있습니다.
void showCelebrationPopup(
  BuildContext context, {
  String titleText = '야호!',
  String subtitleText = '퇴사 질러~~~',
  String characterImagePath = 'assets/images/characters/char_jump.png',
  int confettiDuration = 4000,
  int confettiCount = 20,
  double backgroundOpacity = 0.5, // 새로 추가된 매개변수
}) {
  showDialog(
    context: context,
    barrierDismissible: true,
    barrierColor: Colors.transparent,
    builder: (context) => CelebrationPopup(
      titleText: titleText,
      subtitleText: subtitleText,
      characterImagePath: characterImagePath,
      confettiDuration: confettiDuration,
      confettiCount: confettiCount,
      backgroundOpacity: backgroundOpacity, // 추가된 매개변수
      onClose: () => Navigator.of(context).pop(),
    ),
  );
}

/// 퇴사 축하 팝업을 보여주는 편의 함수 (원래 함수명 유지)
void showRetirementCelebration(BuildContext context) {
  showCelebrationPopup(context,
      titleText: '야호!',
      subtitleText: '퇴사 질러~~~',
      characterImagePath: 'assets/images/characters/char_jump.png',
      backgroundOpacity: 0.7);
}

/*
사용법
기본
showRetirementCelebration(context);

커스터마이즈
showCelebrationPopup(
  context,
  titleText: '야호!',
  subtitleText: '손효자 님의 \n 월급날입니다다',
  characterImagePath: 'assets/images/characters/char_happy.png',
  confettiCount: 20, // 컨페티 개수수
  confettiDuration: 4000, // 애니메이션 지속 시간
  backgroundOpacity: 0.7, // 배경 투명도 조절 (0.0 ~ 1.0)
);

*/
