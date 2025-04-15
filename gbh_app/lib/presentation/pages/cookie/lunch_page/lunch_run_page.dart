import 'package:confetti/confetti.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/core/constants/lunch_menu_data.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/cookie/lunch_page/game/entities/food_ball.dart';
import 'package:marshmellow/presentation/viewmodels/lunch/lunch_view_model.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/router/routes/cookie_routes.dart';
import 'game/lunch_game_widget.dart';
import 'game/lunch_game.dart'; // BoundaryType을 가져오기 위해 추가

class LunchRunPage extends ConsumerStatefulWidget {
  const LunchRunPage({super.key});

  @override
  ConsumerState<LunchRunPage> createState() => _LunchRunPageState();
}

class _LunchRunPageState extends ConsumerState<LunchRunPage> {
  // 게임 위젯 키 - 게임 인스턴스에 접근하기 위해 필요
  final GlobalKey<LunchGameWidgetState> _gameKey = GlobalKey<LunchGameWidgetState>();
  bool _gameStarted = false;
  List<String> _winners = [];
  // 현재 선택된 경기장 타입
  BoundaryType _currentBoundaryType = BoundaryType.DEFAULT;

  // ConfettiController 추가
  late ConfettiController _confettiController;

  @override
  void initState() {
    super.initState();
    // 10초 동안 실행되는 ConfettiController 생성
    _confettiController = ConfettiController(duration: const Duration(seconds: 10));
  }
  
  @override
  void dispose() {
    _confettiController.dispose();
    super.dispose();
  }

  Path drawHeart(Size size) {
    double width = size.width;
    double height = size.height;

    Path path = Path();

    path.moveTo(0.5 * width, height * 0.35);
    path.cubicTo(0.2 * width, height * 0.1, -0.25 * width, height * 0.6,
        0.5 * width, height);
    path.moveTo(0.5 * width, height * 0.35);
    path.cubicTo(0.8 * width, height * 0.1, 1.25 * width, height * 0.6,
        0.5 * width, height);

    path.close();
    return path;
  }

  @override
  Widget build(BuildContext context) {
  // 뷰모델에서 선택된 메뉴 목록 가져오기
  final lunchViewModel = ref.watch(lunchViewModelProvider);
  final selectedMenus = lunchViewModel.selectedMenus;
  
  return Scaffold(
    body: Stack(
      children: [
        // 게임 위젯을 전체 화면으로 배치 (최상위 레이어)
        selectedMenus.isEmpty
          ? const Center(child: Text('선택된 메뉴가 없습니다.'))
          : LunchGameWidget(
              key: _gameKey,
              selectedMenus: selectedMenus,
              onGameComplete: _handleGameComplete,
            ),

        // 상단 앱바 (게임 위에 오버레이)
        // Positioned(
        //   top: 0,
        //   left: 0,
        //   right: 0,
        //   child: SafeArea(
        //     child: Container(
        //       height: 70,
        //       alignment: Alignment.center,
        //       child: Text(
        //         '[ 여기에 Text 입력 ]', 
        //         style: AppTextStyles.mainMoneyTitle.copyWith(color: AppColors.background),
        //       ),
        //     ),
        //   ),
        // ),
        
        // 경기장 타입 선택 드롭다운 (왼쪽 상단)
        // Positioned(
        //   bottom: 60,
        //   left: 10,
        //   child: Container(
        //     padding: EdgeInsets.symmetric(horizontal: 12, vertical: 0),
        //     decoration: BoxDecoration(
        //       color: AppColors.backgroundBlack,
        //       borderRadius: BorderRadius.circular(10),
        //     ),
        //     child: Row(
        //       children: [
        //         SvgPicture.asset(
        //               IconPath.map,
        //               width: 20,
        //               height: 20,
        //               color: AppColors.background,
        //             ),
        //         SizedBox(width: 8),
        //         DropdownButton<BoundaryType>(
        //           value: _currentBoundaryType,
        //           dropdownColor: Colors.grey.shade800,
        //           style: TextStyle(color: Colors.white),
        //           underline: Container(),
        //           onChanged: (newValue) {
        //             if (newValue != null) {
        //               setState(() {
        //                 _currentBoundaryType = newValue;
        //               });
        //               _gameKey.currentState?.changeBoundaryType(newValue);
        //             }
        //           },
        //           items: BoundaryType.values.map((type) {
        //             String label = '';
        //             switch (type) {
        //               case BoundaryType.DEFAULT:
        //                 label = 'Rocio';
        //                 break;
        //               case BoundaryType.ZIGZAG:
        //                 label = 'Viento';
        //                 break;
        //               case BoundaryType.ANGLED:
        //                 label = 'Libera';
        //                 break;
        //               case BoundaryType.CURVED:
        //                 label = 'Somnium';
        //                 break;
        //               case BoundaryType.CIRCULAR:
        //                 label = 'Seio';
        //                 break;
        //               case BoundaryType.CUSTOM:
        //                 label = 'Insula';
        //                 break;
        //             }
        //             return DropdownMenuItem<BoundaryType>(
        //               value: type,
        //               child: Text(label),
        //             );
        //           }).toList(),
        //         ),
        //       ],
        //     ),
        //   ),
        // ),

        // 메뉴 리스트 (오른쪽 상단)
        if (selectedMenus.isNotEmpty)
          Positioned(
            top: 80, // 앱바 아래 위치
            right: 10,
            child: Container(
              width: 60,
              height: selectedMenus.length * 60,
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.4),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: List.generate(
                    selectedMenus.length > 10 ? 10 : selectedMenus.length, 
                    (index) {
                      final menu = selectedMenus[index];
                      return Container(
                        width: 50,
                        height: 50,
                        padding: EdgeInsets.all(8.0),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.asset(
                            menu.imagePath,
                            fit: BoxFit.cover,
                          ),
                        ),
                      );
                    }
                  ),
                ),
              ),
            ),
          ),

        // 하단 버튼 영역
        Positioned(
          bottom: 10,
          left: 0,
          right: 0,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                // 게임 시작/다시하기 버튼
                ElevatedButton(
                  onPressed: _gameStarted ? _resetGame : _startGame,
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.fromLTRB(15.0, 10.0, 15.0, 10.0),
                    backgroundColor: AppColors.buttonBlack,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,  // Row의 크기를 내용물에 맞게 설정
                  mainAxisAlignment: MainAxisAlignment.center,  // 가운데 정렬
                  children: [
                    _gameStarted 
                      ? SvgPicture.asset(
                          IconPath.gas,  // tent 아이콘으로 변경 필요
                          width: 20,
                          height: 20,
                          color: AppColors.background,
                        )
                      : SvgPicture.asset(
                          IconPath.rocket,  // rocket 아이콘으로 변경 필요
                          width: 20,
                          height: 20,
                          color: AppColors.background,
                        ),
                    const SizedBox(width: 8),  // 아이콘과 텍스트 사이 간격
                    Text(
                      _gameStarted ? '다시하기' : '시작하기',
                      style: AppTextStyles.bodySmall.copyWith(color: Colors.grey[200]),
                    ),
                  ],
                ),
                ),
                SizedBox(width: 12),
                // 돌아가기 버튼
                ElevatedButton(
                  onPressed: () {
                    context.replace(CookieRoutes.getLunchPath());
                  },
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.fromLTRB(15.0, 10.0, 15.0, 10.0),
                    backgroundColor: AppColors.buttonBlack,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  child: Row(
                    children: [
                      SvgPicture.asset(
                        IconPath.tent,  // tent 아이콘으로 변경 필요
                        width: 20,
                        height: 20,
                        color: AppColors.background,
                      ),
                      const SizedBox(width: 8),
                      Text('돌아가기', 
                        style: AppTextStyles.bodySmall.copyWith(color: Colors.grey[200]),
                      ),
                    ],
                  ),
                ),
                

              ],
            ),
          ),
        ),

        // 게임 결과 오버레이
        if (_winners.isNotEmpty)
          _buildResultOverlay(),
      ],
    ),
  );
}
  
  // 결과 오버레이 위젯
  Widget _buildResultOverlay() {
    // 우승 메뉴 이름에 맞는 LunchMenu 객체 찾기
    LunchMenu? winnerMenu = allLunchMenus.firstWhere(
      (menu) => menu.name == _winners[0],
      orElse: () => allLunchMenus[0], // 일치하는 메뉴가 없을 경우 기본값 설정
    );
    _confettiController.play();

    return Container(
      color: Colors.black.withOpacity(0.9),
      child: Stack(
        children: [

          Align(
            alignment: Alignment.topCenter,
            child: ConfettiWidget(
              confettiController: _confettiController,
              blastDirectionality: BlastDirectionality.explosive, // 모든 방향으로 폭발하듯 발사
              emissionFrequency: 0.05, // 발사 빈도 (값이 낮을수록 더 많은 조각)
              numberOfParticles: 20, // 한 번에 발사되는 조각 수
              maxBlastForce: 10, // 발사 강도 최대값
              minBlastForce: 5, // 발사 강도 최소값
              gravity: 0.1, // 중력 (낮을수록 천천히 떨어짐)
              shouldLoop: true, // 애니메이션 반복 여부
              colors: const [ // 색상 설정
                Colors.pink,
                Colors.red,
                Colors.orange,
                Colors.purple,
                Colors.blue,
              ],
              createParticlePath: drawHeart, // 하트 모양 함수 사용
            ),
          ),

          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text(
                  '오늘의',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 50,
                    fontWeight: FontWeight.w400,
                  ),
                ),
                const Text(
                  '점심은',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 50,
                    fontWeight: FontWeight.w400,
                  ),
                ),
                const SizedBox(height: 10),
                Text('${_winners[0]}', 
                  style: TextStyle(
                    color: const Color.fromARGB(255, 255, 155, 155),
                    fontSize: 50,
                    fontWeight: FontWeight.w400,
                  ),),
                const SizedBox(height: 10),
                Image.asset(
                winnerMenu.imagePath,
                width: 120,
                height: 120,
                fit: BoxFit.contain,
              ),
                
                const SizedBox(height: 40),
                const Text(
                  '다시 하려면 "다시하기" 버튼을 누르세요',
                  style: TextStyle(
                    color: Colors.white70,
                    fontSize: 14,
                  ),
                ),
              ],
            ),
          ),
                // 하단 버튼 영역
                Positioned(
                  bottom: 10,
                  left: 0,
                  right: 0,
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        // 게임 시작/다시하기 버튼
                        ElevatedButton(
                          onPressed: _gameStarted ? _resetGame : _startGame,
                          style: ElevatedButton.styleFrom(
                            padding: EdgeInsets.fromLTRB(15.0, 10.0, 15.0, 10.0),
                            backgroundColor: AppColors.buttonBlack,
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,  // Row의 크기를 내용물에 맞게 설정
                          mainAxisAlignment: MainAxisAlignment.center,  // 가운데 정렬
                          children: [
                            _gameStarted 
                              ? SvgPicture.asset(
                                  IconPath.gas,  // tent 아이콘으로 변경 필요
                                  width: 20,
                                  height: 20,
                                  color: AppColors.background,
                                )
                              : SvgPicture.asset(
                                  IconPath.rocket,  // rocket 아이콘으로 변경 필요
                                  width: 20,
                                  height: 20,
                                  color: AppColors.background,
                                ),
                            const SizedBox(width: 8),  // 아이콘과 텍스트 사이 간격
                            Text(
                              _gameStarted ? '다시하기' : '시작하기',
                              style: AppTextStyles.bodySmall.copyWith(color: Colors.grey[200]),
                            ),
                          ],
                        ),
                        ),
                        SizedBox(width: 12),
                        // 돌아가기 버튼
                        ElevatedButton(
                          onPressed: () {
                            context.replace(CookieRoutes.getLunchPath());
                          },
                          style: ElevatedButton.styleFrom(
                            padding: EdgeInsets.fromLTRB(15.0, 10.0, 15.0, 10.0),
                            backgroundColor: AppColors.buttonBlack,
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                          child: Row(
                            children: [
                              SvgPicture.asset(
                                IconPath.tent,  // tent 아이콘으로 변경 필요
                                width: 20,
                                height: 20,
                                color: AppColors.background,
                              ),
                              const SizedBox(width: 8),
                              Text('돌아가기', 
                                style: AppTextStyles.bodySmall.copyWith(color: Colors.grey[200]),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
        ],
      ),
    );
  }
  
  // 게임 시작 메서드
  void _startGame() {
    setState(() {
      _gameStarted = true;
      _winners = [];
    });
    _gameKey.currentState?.startGame();
  }
  
  // 게임 리셋 메서드
  void _resetGame() {
    setState(() {
      _gameStarted = false;
      _winners = [];
    });
    _gameKey.currentState?.resetGame();
    _confettiController.stop();
  }
  
  // 게임 결과 처리 콜백
  void _handleGameComplete(List finishedBalls) {
    // dynamic 대신 명시적 형변환 사용
    final winners = finishedBalls.map((ball) => ball.name.toString()).toList();
    setState(() {
      _winners = winners;
    });
    _confettiController.play();
  }
}