import 'package:flame_forge2d/flame_forge2d.dart';
import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'dart:math' as math;

import 'entities/wall.dart';
import 'entities/floor.dart';

class BoundaryManager {
  final Forge2DGame game;
  final Vector2 worldSize;

  BoundaryManager({
    required this.game,
    required this.worldSize,
  });

void addDefaultBoundaries() {
  final wallThickness = worldSize.x * 0.01;

  
  // 바닥 추가
  final floor = Floor(
    position: Vector2(worldSize.x / 2, worldSize.y - 20),
    size: Vector2(worldSize.x, 40),
    color: Colors.white,
  );
  game.add(floor);
  
  // 왼쪽 벽 (기본 경계)
  game.add(Wall(
    position: Vector2(wallThickness / 2, worldSize.y / 2),
    size: Vector2(wallThickness, worldSize.y),
    color: Colors.brown.shade600,
  ));
  
  // 오른쪽 벽 (기본 경계)
  game.add(Wall(
    position: Vector2(worldSize.x - wallThickness / 2, worldSize.y / 2),
    size: Vector2(wallThickness, worldSize.y),
    color: Colors.brown.shade600,
  ));

// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // 상단에 회전하는 막대기 추가
  final rotatingBarWidth = worldSize.x * 0.1; // 막대기 길이는 화면 너비의 10%
  final rotatingBarHeight = worldSize.x * 0.02; // 막대기 높이는 벽 두께의 2배
  final rotatingBarY = worldSize.y * 0.1; // 화면 상단에서 10% 위치에 배치

  // 롤링썬더
  game.add(Wall(
    position: Vector2(worldSize.x / 2, rotatingBarY*1.55),
    size: Vector2(rotatingBarWidth*2, rotatingBarHeight),
    color: AppColors.warnningLight,
    isRotating: true, // Wall 클래스에 isRotating 속성 추가 필요
    rotationSpeed: 2.0, // 회전 속도 설정 (라디안/초)
  ));

  game.add(Wall(
    position: Vector2(worldSize.x *0.8, rotatingBarY*2.25),
    size: Vector2(rotatingBarWidth*3, rotatingBarHeight),
    color: AppColors.warnningLight,
    isRotating: true, // Wall 클래스에 isRotating 속성 추가 필요
    rotationSpeed: -3.0, // 회전 속도 설정 (라디안/초)
  ));
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  final MyX = worldSize.x*0.1; // 장애물 높이 (화면 높이의 1/20)
  final MyY = worldSize.y*0.1; // 장애물 높이 (화면 높이의 1/20)
  final MyWidth = worldSize.x * 0.02; // 장애물 너비 (화면 너비의 60%)
  final MyHeight = worldSize.y * 0.01; // 세로 간격 (화면 높이의 1/12)
  game.add(Wall(
    position: Vector2(MyX*1.3  , MyY*0.3),
    size: Vector2(MyWidth, MyHeight*21),
    color: AppColors.blueDark,
    // angle:  math.pi / 3.14
  ));
  game.add(Wall(
    position: Vector2(MyX*7.3  , MyY*0.3),
    size: Vector2(MyWidth, MyHeight*21),
    color: AppColors.blueDark,
    // angle:  math.pi / 3.14
  ));
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  // 지그재그 장애물 추가 (왼쪽 -> 오른쪽 방향)
  final obstacleHeight = worldSize.y / 20; // 장애물 높이 (화면 높이의 1/20)
  final obstacleWidth = worldSize.x * 0.6; // 장애물 너비 (화면 너비의 60%)
  final verticalSpacing = worldSize.y / 12; // 세로 간격 (화면 높이의 1/12)

    game.add(Wall(
      position: Vector2(MyX*3.8, MyY*1.54),
      size: Vector2(obstacleWidth, wallThickness * 2),
      color: AppColors.blueDark,
      angle:  math.pi / 11
    ));
    game.add(Wall(
      position: Vector2(MyX*9.8, MyY*1.54),
      size: Vector2(obstacleWidth, wallThickness * 2),
      color: AppColors.blueDark,
      angle:  math.pi / 11
    ));
    game.add(Wall(
      position: Vector2(MyX*9.8, MyY*2.04),
      size: Vector2(obstacleWidth, wallThickness * 2),
      color: AppColors.blueDark,
      angle:  -math.pi / 8
    ));




  // 왼쪽에서 시작하는 장애물들
  for (int i = 2; i < 10; i += 2) {
    final yPos = obstacleHeight * 3 + (i * verticalSpacing);
    
    // 왼쪽 시작, 오른쪽으로 뻗는 장애물
    game.add(Wall(
      position: Vector2(worldSize.x * 0.4, yPos),
      size: Vector2(obstacleWidth, wallThickness * 2),
      color: AppColors.pinkPrimary,
      angle:  math.pi / 12
    ));
  }
  
  // 오른쪽에서 시작하는 장애물들
  for (int i = 1; i < 10; i += 2) {
    final yPos = obstacleHeight * 3 + (i * verticalSpacing);
    
    // 오른쪽 시작, 왼쪽으로 뻗는 장애물
    game.add(Wall(
      position: Vector2(worldSize.x * 0.7, yPos),
      size: Vector2(obstacleWidth, wallThickness * 2),
      color: AppColors.blueDark,
      angle:  -math.pi / 10
    ));
  }


    // game.add(Wall(
    //   position: Vector2(worldSize.x * 0.4, worldSize.y *1  ),
    //   size: Vector2(obstacleWidth, wallThickness * 2),
    //   color: AppColors.buttonDelete,
    //   angle:  math.pi / 12
    // ));
    game.add(Wall(
      position: Vector2(worldSize.x * 0.1, worldSize.y *0.95  ),
      size: Vector2(worldSize.x * 0.7, wallThickness * 2),
      color: AppColors.buttonDelete,
      angle:  math.pi / 11
    ));
    game.add(Wall(
      position: Vector2(worldSize.x * 0.9, worldSize.y *0.95  ),
      size: Vector2(worldSize.x * 0.7, wallThickness * 2),
      color: AppColors.buttonDelete,
      angle:  -math.pi / 11
    ));
    game.add(Wall(
      position: Vector2(worldSize.x * 0.2, worldSize.y *0.96 ),
      size: Vector2(rotatingBarWidth*2, rotatingBarHeight),
      color: AppColors.warnningLight,
      isRotating: true, // Wall 클래스에 isRotating 속성 추가 필요
      rotationSpeed: -2.0, // 회전 속도 설정 (라디안/초)
    ));
    game.add(Wall(
      position: Vector2(worldSize.x * 0.8, worldSize.y *0.96),
      size: Vector2(rotatingBarWidth*2, rotatingBarHeight),
      color: AppColors.warnningLight,
      isRotating: true, // Wall 클래스에 isRotating 속성 추가 필요
      rotationSpeed: 2.0, // 회전 속도 설정 (라디안/초)
    ));



  // 세로 방향 짧은 장애물들 (양쪽에 지그재그로 배치)
  final verticalObstacleHeight = worldSize.y / 8; // 세로 장애물 높이
  final verticalObstacleWidth = wallThickness * 2; // 세로 장애물 너비
  
  // 왼쪽 세로 장애물들
  for (int i = 0; i < 8; i += 3) {
    final yPos = obstacleHeight * 5 + (i * verticalSpacing);
    
    game.add(Wall(
      position: Vector2(worldSize.x * 0.2, yPos),
      size: Vector2(verticalObstacleWidth, verticalObstacleHeight),
      color: AppColors.greenPrimary,
    ));
  }
  
  // 오른쪽 세로 장애물들
  // for (int i = 1; i < 10; i += 3) {
  //   final yPos = obstacleHeight * 5 + (i * verticalSpacing);
    
  //   game.add(Wall(
  //     position: Vector2(worldSize.x * 0.8, yPos),
  //     size: Vector2(verticalObstacleWidth, verticalObstacleHeight),
  //     color: AppColors.greenPrimary,
  //   ));
  // }
  game.add(Wall(
    position: Vector2(worldSize.x * 0.8, obstacleHeight * 5 + (1 * verticalSpacing)),
    size: Vector2(verticalObstacleWidth, verticalObstacleHeight),
    color: AppColors.greenPrimary,
  ));
  game.add(Wall(
    position: Vector2(worldSize.x * 0.8, obstacleHeight * 5 + (7 * verticalSpacing)),
    size: Vector2(verticalObstacleWidth, verticalObstacleHeight),
    color: AppColors.greenPrimary,
  ));
  
  
  // 중간에 작은 통로를 만드는 벽들
  final tunnelY = worldSize.y * 0.63; // 통로 y좌표
  
  // 왼쪽 벽
  game.add(Wall(
    position: Vector2(worldSize.x * 0.15, tunnelY),
    size: Vector2(worldSize.x * 0.4, wallThickness * 2),
    color: Colors.deepPurple,
    angle:  math.pi / 20
  ));
  
  // 오른쪽 벽
  game.add(Wall(
    position: Vector2(worldSize.x * 0.85, tunnelY),
    size: Vector2(worldSize.x * 0.4, wallThickness * 2),
    color: Colors.deepPurple,
    angle:  -math.pi / 20
  ));
}}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// 아래는 참고용 코드 
// 아래는 참고용 코드
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

// // BoundaryManager 클래스 내에 있는 addCustomBoundaries 메서드 수정
// void addZigZagBoundaries() {
//   final wallThickness = worldSize.x * 0.015;
  
//   // 바닥 추가
//   final floor = Floor(
//     position: Vector2(worldSize.x / 2, worldSize.y - 20),
//     size: Vector2(worldSize.x, 40),
//     color: Colors.white,
//   );
//   game.add(floor);
  
//   // 통로의 폭 정의
//   final pathWidth = worldSize.x * 0.3; // 통로 폭은 화면 너비의 30%
  
//   // 이미지에 맞게 지그재그 통로 벽 추가 (외부 벽)
//   List<Vector2> leftPathPoints = [];
//   List<Vector2> rightPathPoints = [];
  
//   // 시작점 (상단)
//   leftPathPoints.add(Vector2(worldSize.x * 0.3, 0));
//   rightPathPoints.add(Vector2(worldSize.x * 0.3 + pathWidth, 0));
  
//   // 첫 번째 직선 구간
//   leftPathPoints.add(Vector2(worldSize.x * 0.3, worldSize.y * 0.15));
//   rightPathPoints.add(Vector2(worldSize.x * 0.3 + pathWidth, worldSize.y * 0.15));
  
//   // 첫 번째 꺾임 (우측으로)
//   leftPathPoints.add(Vector2(worldSize.x * 0.5, worldSize.y * 0.25));
//   rightPathPoints.add(Vector2(worldSize.x * 0.5 + pathWidth, worldSize.y * 0.25));
  
//   // 두 번째 직선 구간
//   leftPathPoints.add(Vector2(worldSize.x * 0.5, worldSize.y * 0.35));
//   rightPathPoints.add(Vector2(worldSize.x * 0.5 + pathWidth, worldSize.y * 0.35));
  
//   // 두 번째 꺾임 (좌측으로)
//   leftPathPoints.add(Vector2(worldSize.x * 0.2, worldSize.y * 0.45));
//   rightPathPoints.add(Vector2(worldSize.x * 0.2 + pathWidth, worldSize.y * 0.45));
  
//   // 세 번째 직선 구간
//   leftPathPoints.add(Vector2(worldSize.x * 0.2, worldSize.y * 0.55));
//   rightPathPoints.add(Vector2(worldSize.x * 0.2 + pathWidth, worldSize.y * 0.55));
  
//   // 세 번째 꺾임 (우측으로)
//   leftPathPoints.add(Vector2(worldSize.x * 0.5, worldSize.y * 0.65));
//   rightPathPoints.add(Vector2(worldSize.x * 0.5 + pathWidth, worldSize.y * 0.65));
  
//   // 네 번째 직선 구간 (가운데 우측으로)
//   leftPathPoints.add(Vector2(worldSize.x * 0.5, worldSize.y * 0.75));
//   rightPathPoints.add(Vector2(worldSize.x * 0.5 + pathWidth, worldSize.y * 0.75));
  
//   // 네 번째 꺾임 (다이아몬드 부분 시작)
//   leftPathPoints.add(Vector2(worldSize.x * 0.4, worldSize.y * 0.85));
//   rightPathPoints.add(Vector2(worldSize.x * 0.65, worldSize.y * 0.85));
  
//   // 다이아몬드 하단
//   leftPathPoints.add(Vector2(worldSize.x * 0.2, worldSize.y * 0.95));
//   rightPathPoints.add(Vector2(worldSize.x * 0.5 + pathWidth, worldSize.y * 0.95));
  
//   // 마지막 직선 구간 (바닥까지)
//   leftPathPoints.add(Vector2(worldSize.x * 0.2, worldSize.y - 40));
//   rightPathPoints.add(Vector2(worldSize.x * 0.5 + pathWidth, worldSize.y - 40));
  
//   // 중앙에 다이아몬드 형태의 장애물 추가
//   List<Vector2> diamondPoints = [];
//   diamondPoints.add(Vector2(worldSize.x * 0.4, worldSize.y * 0.75));  // 상단
//   diamondPoints.add(Vector2(worldSize.x * 0.5, worldSize.y * 0.85));  // 우측
//   diamondPoints.add(Vector2(worldSize.x * 0.4, worldSize.y * 0.95));  // 하단
//   diamondPoints.add(Vector2(worldSize.x * 0.3, worldSize.y * 0.85));  // 좌측
  
//   // 각 직선 구간에 대해 벽 추가
//   for (int i = 0; i < leftPathPoints.length - 1; i++) {
//     // 왼쪽 통로 벽
//     _addWallBetweenPoints(leftPathPoints[i], leftPathPoints[i + 1], wallThickness, const Color.fromARGB(255, 61, 255, 35));
    
//     // 오른쪽 통로 벽
//     _addWallBetweenPoints(rightPathPoints[i], rightPathPoints[i + 1], wallThickness, const Color.fromARGB(255, 58, 84, 230));
//   }
  
//   // 다이아몬드 장애물 추가 (4개의 벽)
//   for (int i = 0; i < diamondPoints.length; i++) {
//     final nextIndex = (i + 1) % diamondPoints.length;
//     _addWallBetweenPoints(diamondPoints[i], diamondPoints[nextIndex], wallThickness, Colors.orange);
//   }
  
//   // 좌우 경계 벽 (화면 밖으로 나가지 않도록)
//   // 왼쪽 벽
//   game.add(Wall(
//     position: Vector2(0, worldSize.y / 2),
//     size: Vector2(wallThickness, worldSize.y),
//     color: Colors.blueGrey.shade800,
//   ));
  
//   // 오른쪽 벽
//   game.add(Wall(
//     position: Vector2(worldSize.x, worldSize.y / 2),
//     size: Vector2(wallThickness, worldSize.y),
//     color: Colors.blueGrey.shade800,
//   ));
// }

// // 두 점 사이에 벽 추가하는 헬퍼 메서드
// void _addWallBetweenPoints(Vector2 start, Vector2 end, double thickness, Color color) {
//   // 두 점 사이의 거리 계산
//   final dx = end.x - start.x;
//   final dy = end.y - start.y;
//   final length = math.sqrt(dx * dx + dy * dy);
  
//   // 두 점의 중간점 계산
//   final midX = (start.x + end.x) / 2;
//   final midY = (start.y + end.y) / 2;
  
//   // 회전 각도 계산
//   final angle = math.atan2(dy, dx);
  
//   // 벽 생성
//   game.add(Wall(
//     position: Vector2(midX, midY),
//     size: Vector2(length, thickness),
//     angle: angle,
//     color: color,
//   ));
// }

// // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

//   // 경사진 경계 추가
//   void addAngledBoundaries() {
//     final wallThickness = worldSize.x * 0.01;
//     final wallLength = worldSize.y / 3; // 세 개의 세그먼트로 나누기
    
//     // 바닥 추가
//     final floor = Floor(
//       position: Vector2(worldSize.x / 2, worldSize.y - 20),
//       size: Vector2(worldSize.x, 40),
//       color: Colors.white,
//     );
//     game.add(floor);
    
//     // 왼쪽 경사 벽
//     for (int i = 0; i < 3; i++) {
//       final yPos = i * wallLength + wallLength / 2;
//       final angle = (i % 2 == 0) ? math.pi / 12 : -math.pi / 12; // +/- 15도
      
//       game.add(Wall(
//         position: Vector2(wallThickness * 5, yPos),
//         size: Vector2(wallThickness, wallLength),
//         angle: angle,
//         color: Colors.brown.shade600,
//       ));
//     }
    
//     // 오른쪽 경사 벽
//     for (int i = 0; i < 3; i++) {
//       final yPos = i * wallLength + wallLength / 2;
//       final angle = (i % 2 == 0) ? -math.pi / 12 : math.pi / 12; // +/- 15도
      
//       game.add(Wall(
//         position: Vector2(worldSize.x - wallThickness * 5, yPos),
//         size: Vector2(wallThickness, wallLength),
//         angle: angle,
//         color: Colors.brown.shade600,
//       ));
//     }
//   }


// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

//   // 곡선 경계 추가
//   void addCurvedBoundaries() {
//     // 바닥 추가
//     final floor = Floor(
//       position: Vector2(worldSize.x / 2, worldSize.y - 20),
//       size: Vector2(worldSize.x, 40),
//       color: Colors.white,
//     );
//     game.add(floor);
    
//     // 왼쪽 곡선 벽 - 사인파 형태
//     final leftCurvePoints = <Vector2>[];
//     for (int i = 0; i <= 100; i++) {
//       final t = i / 100;
//       final y = t * worldSize.y;
//       final x = math.sin(t * math.pi * 4) * worldSize.x * 0.05 + worldSize.x * 0.1;
//       leftCurvePoints.add(Vector2(x, y));
//     }
    
//     game.add(CurvedWall(
//       position: Vector2.zero(),
//       points: leftCurvePoints,
//       thickness: 5.0,
//       color: Colors.brown.shade600,
//     ));
    
//     // 오른쪽 곡선 벽 - 사인파 형태
//     final rightCurvePoints = <Vector2>[];
//     for (int i = 0; i <= 100; i++) {
//       final t = i / 100;
//       final y = t * worldSize.y;
//       final x = -math.sin(t * math.pi * 4) * worldSize.x * 0.05 + worldSize.x * 0.9;
//       rightCurvePoints.add(Vector2(x, y));
//     }
    
//     game.add(CurvedWall(
//       position: Vector2.zero(),
//       points: rightCurvePoints,
//       thickness: 5.0,
//       color: Colors.brown.shade600,
//     ));
//   }
// // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

//   // 원형 장애물이 있는 경계 추가
//   void addCircularObstacleBoundaries() {
//     // 기본 직선 경계 추가
//     // addDefaultBoundaries();
    
//     // 원형 장애물 추가
//     final obsRadius = worldSize.x * 0.05;
    
//     // 왼쪽 벽 근처 원형 장애물
//     for (int i = 1; i <= 5; i++) {
//       final yPos = worldSize.y * i / 6;
//       game.add(CircularWall(
//         position: Vector2(worldSize.x * 0.2, yPos),
//         radius: obsRadius,
//         color: Colors.orange,
//       ));
//     }
    
//     // 오른쪽 벽 근처 원형 장애물
//     for (int i = 1; i <= 5; i++) {
//       final yPos = worldSize.y * i / 6;
//       game.add(CircularWall(
//         position: Vector2(worldSize.x * 0.8, yPos),
//         radius: obsRadius,
//         color: Colors.orange,
//       ));
//     }
//   }
// // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

//   // 테스트 경기장
//   void testaddCustomBoundaries() {
//     final wallThickness = worldSize.x * 0.01;
    
//     // 바닥 추가
//     final floor = Floor(
//       position: Vector2(worldSize.x / 2, worldSize.y - 20),
//       size: Vector2(worldSize.x, 40),
//       color: Colors.white,
//     );
//     game.add(floor);
    
//     // 미로 같은 구조 만들기
//     // 왼쪽 벽
//     game.add(Wall(
//       position: Vector2(wallThickness / 2, worldSize.y / 2),
//       size: Vector2(wallThickness, worldSize.y),
//       color: Colors.brown.shade600,
//     ));
    
//     // 오른쪽 벽
//     game.add(Wall(
//       position: Vector2(worldSize.x - wallThickness / 2, worldSize.y / 2),
//       size: Vector2(wallThickness, worldSize.y),
//       color: Colors.brown.shade600,
//     ));
    
//     // 가로 장애물들
//     for (int i = 1; i <= 6; i++) {
//       final yPos = worldSize.y * i / 7;
//       final xPos = (i % 2 == 0) ? worldSize.x * 0.25 : worldSize.x * 0.75;
//       final width = worldSize.x * 0.5;
      
//       game.add(Wall(
//         position: Vector2(xPos, yPos),
//         size: Vector2(width, wallThickness * 2),
//         color: Colors.brown.shade400,
//       ));
//     }
    
//     // 곡선 추가
//     final curvePoints = <Vector2>[];
//     for (int i = 0; i <= 50; i++) {
//       final t = i / 50;
//       final angle = t * math.pi;
//       final radius = worldSize.x * 0.15;
//       final x = math.cos(angle) * radius + worldSize.x / 2;
//       final y = math.sin(angle) * radius + worldSize.y / 3;
//       curvePoints.add(Vector2(x, y));
//     }
    
//     game.add(CurvedWall(
//       position: Vector2.zero(),
//       points: curvePoints,
//       thickness: 4.0,
//       color: Colors.purple,
//     ));
    
//     // 원형 장애물
//     game.add(CircularWall(
//       position: Vector2(worldSize.x / 2, worldSize.y * 2/3),
//       radius: worldSize.x * 0.08,
//       color: Colors.teal,
//     ));
//   }

// // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//   // 사용자 정의 커스텀 경계 추가 (대규모 맵 버전)
//   void addCustomBoundaries() {
//     final wallThickness = worldSize.x * 0.01;
    
//     // 바닥 추가 (결승선)
//     final floor = Floor(
//       position: Vector2(worldSize.x / 2, worldSize.y - 20),
//       size: Vector2(worldSize.x, 40),
//       color: Colors.white,
//     );
//     game.add(floor);
    
//     // 측면 벽 대신 곡선과 다양한 형태의 경계 추가
    
//     // 1. 상단 영역 - 랜덤한 곡선 장애물
//     for (int i = 0; i < 5; i++) {
//       final yPos = worldSize.y * (i * 0.05 + 0.05); // 상단 25% 영역
//       final amplitude = worldSize.x * 0.1; // 진폭
//       final frequency = math.pi * (i % 3 + 1) * 0.5; // 다양한 주파수
      
//       final curvePoints = <Vector2>[];
//       for (int j = 0; j <= 50; j++) {
//         final t = j / 50;
//         final x = t * worldSize.x;
//         final y = yPos + math.sin(t * frequency) * amplitude;
//         curvePoints.add(Vector2(x, y));
//       }
      
//       game.add(CurvedWall(
//         position: Vector2.zero(),
//         points: curvePoints,
//         thickness: 4.0,
//         color: Colors.primaries[i % Colors.primaries.length],
//       ));
//     }
    
//     // 2. 중간 영역 - 나선형 장애물
//     final spiralPoints = <Vector2>[];
//     final spiralCenterX = worldSize.x * 0.5;
//     final spiralCenterY = worldSize.y * 0.4;
//     final spiralRadius = worldSize.x * 0.4;
    
//     for (int i = 0; i <= 200; i++) {
//       final t = i / 200;
//       final angle = t * math.pi * 8; // 여러 번 회전
//       final radius = t * spiralRadius;
//       final x = math.cos(angle) * radius + spiralCenterX;
//       final y = math.sin(angle) * radius + spiralCenterY;
//       spiralPoints.add(Vector2(x, y));
//     }
    
//     game.add(CurvedWall(
//       position: Vector2.zero(),
//       points: spiralPoints,
//       thickness: 3.0,
//       color: Colors.purpleAccent,
//     ));
    
//     // 3. 하단 영역 (60~90%) - 지그재그 패턴
//     for (int i = 0; i < 5; i++) {
//       final yPercent = 0.6 + (i * 0.06);
//       final yPos = worldSize.y * yPercent;
//       final zigzagPoints = <Vector2>[];
      
//       for (int j = 0; j <= 20; j++) {
//         final xPercent = j / 20.0;
//         final x = xPercent * worldSize.x;
//         final yOffset = (j % 2 == 0) ? -worldSize.y * 0.02 : worldSize.y * 0.02;
//         zigzagPoints.add(Vector2(x, yPos + yOffset));
//       }
      
//       game.add(CurvedWall(
//         position: Vector2.zero(),
//         points: zigzagPoints,
//         thickness: 4.0,
//         color: Colors.orange,
//       ));
//     }
    
//     // 4. 최하단 영역 (90~100%) - 깔때기 모양 (중앙으로 모이게)
//     final funnelTop = worldSize.y * 0.9; // 깔때기 시작 지점
//     final funnelBottom = worldSize.y - 60; // 깔때기 끝 지점 (바닥 바로 위)
    
//     // 왼쪽 깔때기 경사면
//     final leftFunnelPoints = <Vector2>[];
//     leftFunnelPoints.add(Vector2(0, funnelTop)); // 왼쪽 끝에서 시작
//     leftFunnelPoints.add(Vector2(worldSize.x * 0.4, funnelBottom)); // 중앙 근처로 모임
    
//     game.add(CurvedWall(
//       position: Vector2.zero(),
//       points: leftFunnelPoints,
//       thickness: 5.0,
//       color: Colors.red,
//     ));
    
//     // 오른쪽 깔때기 경사면
//     final rightFunnelPoints = <Vector2>[];
//     rightFunnelPoints.add(Vector2(worldSize.x, funnelTop)); // 오른쪽 끝에서 시작
//     rightFunnelPoints.add(Vector2(worldSize.x * 0.6, funnelBottom)); // 중앙 근처로 모임
    
//     game.add(CurvedWall(
//       position: Vector2.zero(),
//       points: rightFunnelPoints,
//       thickness: 5.0,
//       color: Colors.red,
//     ));
    
//     // 5. 중간에 랜덤 원형 장애물 추가
//     final random = math.Random(42); // 시드 설정하여 매번 같은 난수 생성
//     for (int i = 0; i < 15; i++) {
//       final x = worldSize.x * (0.1 + random.nextDouble() * 0.8); // 좌우 여백 10%씩
//       final y = worldSize.y * (0.1 + random.nextDouble() * 0.7); // 상단 10%, 하단 20% 제외
//       final radius = worldSize.x * (0.02 + random.nextDouble() * 0.03); // 다양한 크기
      
//       // 원형 장애물 색상 다양하게
//       final colorIndex = random.nextInt(Colors.primaries.length);
      
//       game.add(CircularWall(
//         position: Vector2(x, y),
//         radius: radius,
//         color: Colors.primaries[colorIndex].withOpacity(0.8),
//       ));
//     }
//   }
// // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



