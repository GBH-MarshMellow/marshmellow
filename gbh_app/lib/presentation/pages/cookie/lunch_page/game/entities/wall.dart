import 'package:flame_forge2d/flame_forge2d.dart';
import 'package:flutter/material.dart';
import 'base_body.dart';
import 'dart:math' as math;

class Wall extends BaseBody {
  final Vector2 size;
  final double angle; // 회전 각도 추가 (라디안)
  bool isRotating;
  double rotationSpeed;

  Wall({
    required Vector2 position,
    required this.size,
    this.angle = 0.0, // 기본값 0 (회전 없음)
    Color color = Colors.brown,
    this.isRotating = false,
    this.rotationSpeed = 0.0,
  }) : super(position: position, color: color);

  @override
  Body createBody() {
    // 벽은 움직이지 않는 정적 바디로 생성
    final bodyDef = BodyDef(
      position: position,
      type: BodyType.static,
      angle: angle, // 회전 각도 적용
    );

    // 바디 생성
    final body = world.createBody(bodyDef);

    // 사각형 모양의 픽스처 생성
    final shape = PolygonShape()
      ..setAsBox(
        size.x / 2, // 너비의 절반
        size.y / 2, // 높이의 절반
        Vector2(0, 0), // 중심점
        0, // shape의 회전각도는 0으로 두고 body에서 회전 처리
      );

    // 픽스처 속성 설정
    final fixtureDef = FixtureDef(shape)
      ..restitution = 0.1 // 낮은 반발력
      ..friction = 0.5;   // 높은 마찰력

    // 바디에 픽스처 추가
    body.createFixture(fixtureDef);

    return body;
  }

  @override
  void update(double dt) {
    super.update(dt);
    
    if (isRotating) {
      // 시간에 따라 회전 각도 업데이트
      body.setTransform(body.position, body.angle + rotationSpeed * dt);
    }
  }

  @override
  void render(Canvas canvas) {
    canvas.save();
    
    // 회전 적용
    canvas.rotate(body.angle);
    
    // 벽 시각적으로 그리기
    final rect = Rect.fromCenter(
      center: Offset(0, 0),
      width: size.x,
      height: size.y,
    );

    canvas.drawRect(
      rect,
      Paint()..color = color,
    );
    
    canvas.restore();
  }
}

// 곡선 벽 클래스 추가
class CurvedWall extends BaseBody {
  final List<Vector2> points; // 곡선을 구성하는 점들
  final double thickness; // 선의 두께
  
  CurvedWall({
    required Vector2 position,
    required this.points,
    this.thickness = 5.0,
    Color color = Colors.brown,
  }) : super(position: position, color: color);

  @override
  Body createBody() {
    final bodyDef = BodyDef(
      position: position,
      type: BodyType.static,
    );

    final body = world.createBody(bodyDef);

    // ChainShape 사용하여 곡선 생성
    final shape = ChainShape()
      ..createChain(points);
    
    final fixtureDef = FixtureDef(shape)
      ..restitution = 0.1
      ..friction = 0.5;
    
    body.createFixture(fixtureDef);
    
    return body;
  }



  @override
  void render(Canvas canvas) {
    final path = Path();
    
    if (points.isEmpty) return;
    
    // 시작점 설정
    path.moveTo(points.first.x, points.first.y);
    
    // 나머지 점들을 연결
    for (int i = 1; i < points.length; i++) {
      path.lineTo(points[i].x, points[i].y);
    }
    
    // 경로 그리기
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..strokeWidth = thickness
        ..style = PaintingStyle.stroke,
    );
  }
}

// 원형 벽 클래스 추가
class CircularWall extends BaseBody {
  final double radius;
  
  CircularWall({
    required Vector2 position,
    required this.radius,
    Color color = Colors.brown,
  }) : super(position: position, color: color);

  @override
  Body createBody() {
    final bodyDef = BodyDef(
      position: position,
      type: BodyType.static,
    );

    final body = world.createBody(bodyDef);

    // 원형 모양 생성
    final shape = CircleShape()..radius = radius;
    
    final fixtureDef = FixtureDef(shape)
      ..restitution = 0.1
      ..friction = 0.5;
    
    body.createFixture(fixtureDef);
    
    return body;
  }

  @override
  void render(Canvas canvas) {
    canvas.drawCircle(
      Offset(0, 0),
      radius,
      Paint()..color = color,
    );
  }
}