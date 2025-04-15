import 'package:flame_forge2d/flame_forge2d.dart';
import 'package:flutter/material.dart';

// 모든 게임 객체의 기본이 되는
abstract class BaseBody extends BodyComponent {
  final Vector2 position;
  final Color color;

  BaseBody({
    required this.position,
    this.color = Colors.white,
  });

  @override
  Body createBody() {
    // 각 하위 클래스에서 구현
    throw UnimplementedError();
  }
}