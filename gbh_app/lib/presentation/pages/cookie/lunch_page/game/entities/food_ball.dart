import 'package:flame_forge2d/flame_forge2d.dart';
import 'package:flutter/material.dart';
import 'package:marshmellow/presentation/pages/cookie/lunch_page/game/entities/floor.dart';
import 'package:marshmellow/presentation/pages/cookie/lunch_page/game/entities/wall.dart';
import 'package:marshmellow/presentation/pages/cookie/lunch_page/game/lunch_game.dart';
import 'base_body.dart';
import 'package:flame/sprite.dart'; // Sprite 임포트 추가

class FoodBall extends BaseBody with ContactCallbacks {
  final double radius;
  final String name;
  final String imagePath;
  final LunchGame game; // 추가
  bool _activated = false;
  Sprite? _sprite; // 스프라이트 객체 추가
  bool _imageLoaded = false; // 이미지 로드 상태 추적

  FoodBall({
    required Vector2 position,
    required this.radius,
    required this.name,
    required this.imagePath,
    required this.game, // 추가
    Color color = Colors.red,
  }) : super(position: position, color: color) {
    _loadImage();
  }
    // 이미지 로드 메서드 추가
  Future<void> _loadImage() async {
    try {
      print('원본 이미지 경로: $imagePath');
      
      // 실제 앱 번들 내 이미지 경로
      final path = imagePath.replaceAll('assets/images/', '');
      print('수정된 이미지 경로: $path');
      
      _sprite = await Sprite.load(path);
      _imageLoaded = true;
      print('이미지 로드 성공: $path');
    } catch (e) {
      print('이미지 로드 실패: $e');
      _imageLoaded = false;
    }
  }
  // 공 활성화 메서드
  void activate() {
    if (!_activated) {
      _activated = true;
      // 바디 타입을 동적으로 변경하여 중력 영향을 받도록 함
      body.setType(BodyType.dynamic);
      print('Ball ${name} activated, type: ${body.bodyType}');
    }
  }
  @override
  Body createBody() {
    // 동적(움직이는) 바디 정의
    final bodyDef = BodyDef(
      position: position, 
      type: BodyType.static, // 중력과 힘의 영향을 받음
      userData: this, // 이 객체 참조 저장 (충돌 감지에 사용)
    );

    // 바디 생성
    final body = world.createBody(bodyDef);

    // 원형 모양 정의
    final shape = CircleShape()..radius = radius;

    // 픽스처 속성 설정
    final fixtureDef = FixtureDef(shape)
      ..restitution = 0.5 // 높은 반발력
      ..density = 1.0     // 밀도
      ..friction = 0.2;    // 약간의 마찰력

    // 바디에 픽스처 추가
    body.createFixture(fixtureDef);

    return body;
  }
  @override
  void beginContact(Object? other, Contact contact) {
    print('FoodBall ${name} beginContact called');
    if (other is Floor) {
      print('FoodBall ${name} hit the floor!');
      game.onBallFinished(this); // LunchGame에 알리기
      // 여기서 직접 처리하거나 LunchGame에 알리기
    } else if (other is Wall) {
      print('FoodBall ${name} hit the wall!');
    }
  }

  @override
  void render(Canvas canvas) {
    if (_imageLoaded && _sprite != null) {
      // 이미지가 로드됐으면 스프라이트 렌더링
      final size = Vector2(radius * 2, radius * 2 );
      final position = Vector2(-radius, -radius); // 중앙에 맞추기 위해 오프셋 조정
      
      // 스프라이트 그리기
      _sprite!.render(
        canvas,
        position: position,
        size: size,
      );
    } else {
      // 이미지 로드 실패 또는 로드 전이면 색상으로 표시
      canvas.drawCircle(
        Offset.zero,
        radius,
        Paint()..color = color,
      );
    }
  }void reset() {
  body.setTransform(position, 0); // 초기 위치로
  body.setType(BodyType.static); // 정적 상태로
}
}