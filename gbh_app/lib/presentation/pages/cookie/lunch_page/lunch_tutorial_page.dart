import 'package:flutter/material.dart';
import 'package:flame/game.dart';
import 'package:flame/events.dart';
import 'package:flame_forge2d/flame_forge2d.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';

class LunchTutorialPage extends StatelessWidget {
  const LunchTutorialPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppbar(title: '테스트'),
      body: GameWidget(
        game: BallGame(),
      ),
    );
  }
}

class BallGame extends Forge2DGame with TapDetector {
  late Ball ball;
  
  BallGame() : super(gravity: Vector2(0, 10.0));
  
  @override
  Future<void> onLoad() async {
    await super.onLoad();
    
    // 경계 생성
    addAll([
      Wall(Vector2(0, size.y), Vector2(size.x, size.y + 5)), // 바닥
      Wall(Vector2(0, 0), Vector2(0 - 5, size.y)), // 왼쪽 벽
      Wall(Vector2(size.x, 0), Vector2(size.x + 5, size.y)), // 오른쪽 벽
    ]);
    
    // 공 생성 및 중앙에 배치
    ball = Ball(size / 2);
    world.add(ball);
    
    // 카메라가 공을 따라가도록 설정
    camera.follow(ball);
  }
  
  @override
  void onTap() {
    super.onTap();
    // 탭했을 때 공에 위쪽 방향 힘 가하기
    ball.body.applyLinearImpulse(Vector2(0, -20));
  }
}

class Ball extends BodyComponent {
  final Vector2 position;
  final double radius = 1.0;
  
  Ball(this.position);
  
  @override
  Body createBody() {
    final bodyDef = BodyDef(
      type: BodyType.dynamic,
      position: position,
    );
    
    final ball = world.createBody(bodyDef);
    
    final shape = CircleShape()..radius = radius;
    final fixtureDef = FixtureDef(
      shape,
      restitution: 0.8, // 튕김 정도
      density: 1.0,
      friction: 0.4,
    );
    
    ball.createFixture(fixtureDef);
    return ball;
  }
  
  @override
  void render(Canvas canvas) {
    super.render(canvas);
    final paint = Paint()..color = Colors.red;
    canvas.drawCircle(Offset.zero, radius, paint);
  }
}

class Wall extends BodyComponent {
  final Vector2 start;
  final Vector2 end;
  
  Wall(this.start, this.end);
  
  @override
  Body createBody() {
    final bodyDef = BodyDef(
      type: BodyType.static,
      position: Vector2.zero(),
    );
    
    final wall = world.createBody(bodyDef);
    
    final shape = EdgeShape()
      ..set(start, end);
    
    wall.createFixture(FixtureDef(shape));
    return wall;
  }
}