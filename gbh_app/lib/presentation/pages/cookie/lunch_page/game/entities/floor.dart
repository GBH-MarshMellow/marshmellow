import 'package:flame_forge2d/flame_forge2d.dart';
import 'package:flutter/material.dart';
import 'base_body.dart';

class Floor extends BaseBody {
  final Vector2 size;

  Floor({
    required Vector2 position,
    required this.size,
    Color color = Colors.red,
  }) : super(position: position, color: color);

  @override
  Body createBody() {
    // 바닥은 움직이지 않는 정적 바디로 생성
    final bodyDef = BodyDef(
      position: position,
      type: BodyType.static,
      userData: this, // Floor 객체 자신을 넣어
    );
    final body = world.createBody(bodyDef); // 바디 생성
    final shape = PolygonShape() // 사각형 모양의 픽스처 생성
      ..setAsBox(
        size.x / 2, // 너비의 절반
        size.y / 2, // 높이의 절반
        Vector2(0, 0), // 중심점 (0,0)
        0, // 회전각도 (라디안)
      );
    final fixtureDef = FixtureDef(shape) // 픽스처 속성 설정
      ..restitution = 0.0 // 반발력 (탄성)
      ..friction = 0.5;   // 마찰력
    body.createFixture(fixtureDef); // 바디에 픽스처 추가
    print('Floor created at ${position.y}');
    return body;
  }

  @override
  void render(Canvas canvas) {
    // 바닥 시각적으로 그리기
    final rect = Rect.fromCenter(
      center: Offset(0, 0),
      width: size.x,
      height: size.y,
    );

    canvas.drawRect(
      rect,
      Paint()..color = color,
    );
  }
}