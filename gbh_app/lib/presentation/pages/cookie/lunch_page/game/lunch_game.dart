import 'dart:ui';

import 'package:flame/camera.dart';
import 'package:flame/game.dart';
import 'package:flame_forge2d/flame_forge2d.dart';
import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'dart:math' as math;

import 'entities/floor.dart';
import 'entities/food_ball.dart';
import 'entities/wall.dart';
import 'boundary_manager.dart'; // 새로 추가한 BoundaryManager 임포트

// 경기장 타입 설정을 위한 열거형 (클래스 외부로 이동)
enum BoundaryType {
  DEFAULT,
  ZIGZAG,
  ANGLED,
  CURVED,
  CIRCULAR,
  CUSTOM
}

class LunchGame extends Forge2DGame {
  final List selectedMenus;
  bool gameStarted = false;
  final List<FoodBall> foodBalls = [];
  final List<FoodBall> finishedBalls = [];
  Function(List<FoodBall>)? onGameComplete;

  FoodBall? _trackedBall;
  late Vector2 worldSize;
  late BoundaryManager boundaryManager; // BoundaryManager 추가

  // 현재 선택된 경기장 타입
  BoundaryType currentBoundaryType = BoundaryType.ZIGZAG;

  // 화면 스크롤링을 위한 변수들
  double _worldYOffset = 0;  // 월드의 y축 오프셋
  double _targetYOffset = 0; // 목표 y축 오프셋
  final double _scrollSpeed = 5.0; // 스크롤 속도 조절 (필요에 따라 조정)

  LunchGame({
    required this.selectedMenus,
    BoundaryType boundaryType = BoundaryType.ZIGZAG, // 기본값 설정
  }) : super(gravity: Vector2(0, 40.0)) {
    currentBoundaryType = boundaryType;
    print('World initialized: ${world != null}');
  }

  @override
  Future<void> onLoad() async {
    await super.onLoad();

    // 월드 크기 설정 (화면 높이의 5배)
    worldSize = Vector2(size.x, size.y * 4);
    
    // BoundaryManager 초기화
    boundaryManager = BoundaryManager(game: this, worldSize: worldSize);
    
    // 경계 추가
    _addBoundaries();

    // 장애물 추가
    // _addObstacles(worldSize);

    // 공 추가
    _addFoodBalls(worldSize);
  }

  // 경계 추가 메서드 (BoundaryManager로 위임)
  void _addBoundaries() {
    switch (currentBoundaryType) {
      // case BoundaryType.ZIGZAG:
      //   boundaryManager.addZigZagBoundaries();
      //   break;
      // case BoundaryType.ANGLED:
      //   boundaryManager.addAngledBoundaries();
      //   break;
      // case BoundaryType.CURVED:
      //   boundaryManager.addCurvedBoundaries();
      //   break;
      // case BoundaryType.CIRCULAR:
      //   boundaryManager.addCircularObstacleBoundaries();
      //   break;
      // case BoundaryType.CUSTOM:
      //   boundaryManager.addCustomBoundaries();
      //   break;
      // case BoundaryType.DEFAULT:
      default:
        boundaryManager.addDefaultBoundaries();
        break;
    }
  }

  // 경기장 타입 변경 메서드 (게임 재시작 필요)
  void changeBoundaryType(BoundaryType newType) {
    currentBoundaryType = newType;
    resetGame();
    // 기존 벽과 바닥 제거 로직 필요 (복잡함)
    // 현재 상태에서는 완전히 게임을 재시작해야 함
  }

  @override
  void update(double dt) {
    super.update(dt * 3.0);
    
    if (gameStarted) {
      // 가장 아래에 있는 공 찾기
      if (foodBalls.isNotEmpty) {
        FoodBall lowestBall = foodBalls.reduce((current, next) => 
          current.body.position.y > next.body.position.y ? current : next);
        _trackedBall = lowestBall;
      }
      
      if (_trackedBall != null) {
        // 공의 위치 가져오기
        final ballPosition = _trackedBall!.body.position;
        
        // 목표 오프셋 계산: 공이 화면의 1/3 지점에 위치하도록
        _targetYOffset = ballPosition.y - (size.y / 3);
        
        // 현재 오프셋을 목표 오프셋으로 부드럽게 이동
        _worldYOffset += (_targetYOffset - _worldYOffset) * dt * _scrollSpeed;
        
        // 디버깅
        print('공 위치: $ballPosition, 오프셋: $_worldYOffset');
        
        // 공이 바닥에 닿았는지 확인
        if (ballPosition.y > worldSize.y - 60 && !finishedBalls.contains(_trackedBall)) {
          onBallFinished(_trackedBall!);
        }
      }
    }
  }

  // 객체의 렌더링 위치를 오프셋만큼 조정
  @override
  void render(Canvas canvas) {
    // 캔버스 상태 저장
    canvas.save();
    
    // 캔버스를 오프셋만큼 이동 (화면 스크롤링 효과)
    canvas.translate(0, -_worldYOffset);
    
    // 배경 그리기
    canvas.drawColor(AppColors.backgroundBlack, BlendMode.src);
    
    // 월드 렌더링
    super.render(canvas);
    
    // 캔버스 상태 복원
    canvas.restore();
  }

  // 장애물 추가 메서드 (변경 없음)
  void _addObstacles(Vector2 worldSize) {
    final obstacleWidth = worldSize.x * 0.15;
    final obstacleHeight = worldSize.y * 0.01;
    
    add(Wall(
      position: Vector2(worldSize.x * 0.3, worldSize.y * 0.3),
      size: Vector2(obstacleWidth, obstacleHeight),
      color: AppColors.whitePrimary,
    ));
    
    add(Wall(
      position: Vector2(worldSize.x * 0.7, worldSize.y * 0.5),
      size: Vector2(obstacleWidth, obstacleHeight),
      color: AppColors.whitePrimary,
    ));
    
    add(Wall(
      position: Vector2(worldSize.x * 0.4, worldSize.y * 0.7),
      size: Vector2(obstacleWidth, obstacleHeight),
      color: AppColors.whitePrimary,
    ));
    
    add(Wall(
      position: Vector2(worldSize.x * 0.6, worldSize.y * 0.8),
      size: Vector2(obstacleWidth, obstacleHeight),
      color: AppColors.whitePrimary,
    ));
  }

// 음식 공 추가 메서드
void _addFoodBalls(Vector2 worldSize) {
  if (selectedMenus.isEmpty) return;
  
  final ballRadius = worldSize.x * 0.04;
  final startY = ballRadius *  13;
  final menuCount = selectedMenus.length;
  
  // 두 줄로 나누기
  final firstRowCount = (menuCount + 1) ~/ 2; // 첫 줄 공 개수 (올림 나눗셈)
  final secondRowCount = menuCount - firstRowCount; // 둘째 줄 공 개수
  
  // 첫 줄 간격 계산
  final firstRowSpacing = worldSize.x / (firstRowCount + 4);
  
  // 둘째 줄 간격 계산
  final secondRowSpacing = secondRowCount > 0 ? worldSize.x / (secondRowCount + 4) : 0;
  
  // 첫 줄 공 추가
  for (int i = 0; i < firstRowCount; i++) {
    final menu = selectedMenus[i];
    final xPos = firstRowSpacing * (i + 2);
    
    final colors = [
      Colors.red, Colors.blue, Colors.green,
      Colors.yellow, Colors.purple, Colors.orange
    ];
    final color = colors[i % colors.length];
    
    final foodBall = FoodBall(
      position: Vector2(xPos, startY),
      radius: ballRadius,
      name: menu.name,
      imagePath: menu.imagePath,
      game: this,
      color: color,
    );
    
    foodBalls.add(foodBall);
    add(foodBall);
  }
  
  // 둘째 줄 공 추가
  for (int i = 0; i < secondRowCount; i++) {
    final menu = selectedMenus[firstRowCount + i];
    final xPos = secondRowSpacing * (i + 2);
    
    final colors = [
      Colors.red, Colors.blue, Colors.green,
      Colors.yellow, Colors.purple, Colors.orange
    ];
    final color = colors[(firstRowCount + i) % colors.length];
    
    final foodBall = FoodBall(
      position: Vector2(xPos.toDouble(), startY + ballRadius * 2.5), // Y위치를 조금 아래로
      radius: ballRadius,
      name: menu.name,
      imagePath: menu.imagePath,
      game: this,
      color: color,
    );
    
    foodBalls.add(foodBall);
    add(foodBall);
  }
}



  // 공이 바닥에 닿았을 때 호출
  void onBallFinished(FoodBall ball) {
    print('Ball finished: ${ball.name}, Position: ${finishedBalls.length}');
    if (!finishedBalls.contains(ball)) {
      finishedBalls.add(ball);
      if (finishedBalls.length == 1) { // 첫 번째 공만 처리
        _finishGame();
      }
    }
  }

  // 게임 종료
  void _finishGame() {
    print('Game finished with winner: ${finishedBalls.first.name}');
    if (onGameComplete != null) {
      onGameComplete!(finishedBalls);
    }
  }

  // 게임 시작
  void startGame() {
    if (gameStarted) return;
    gameStarted = true;
    print('Game started: $gameStarted');
    
    // 월드 오프셋 초기화
    _worldYOffset = 0;
    _targetYOffset = 0;
    
    // 모든 공을 활성화
    for (final ball in foodBalls) {
      ball.activate();
    }
    
    // 추적할 공 선택 (첫 번째 공으로 설정)
    if (foodBalls.isNotEmpty) {
      _trackedBall = foodBalls.first;
      print('Tracked ball set: ${_trackedBall?.name}, position: ${_trackedBall?.body.position}');
    } else {
      print('No food balls to track');
    }
  }

  // 게임 리셋
  void resetGame() {
    gameStarted = false;
    finishedBalls.clear();
    
    // 월드 오프셋 초기화
    _worldYOffset = 0;
    _targetYOffset = 0;
    
    // 공 리셋
    for (final ball in foodBalls) {
      ball.reset();
    }
    
    // 추적 대상 재설정
    if (foodBalls.isNotEmpty) {
      _trackedBall = foodBalls.first;
    }
  }
}