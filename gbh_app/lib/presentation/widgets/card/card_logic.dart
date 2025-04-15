import 'package:flutter/material.dart';

class CardState {
  final double? width;
  final double? height;
  final Color backgroundColor;
  final double borderRadius;
  final EdgeInsetsGeometry padding;

  CardState({
    this.width,
    this.height,
    required this.backgroundColor,
    required this.borderRadius,
    required this.padding,
  });
}

class CardLogic {
  static CardState getCardState({
    required double? width,
    required double? height,
    required Color backgroundColor,
    required double borderRadius,
    required EdgeInsetsGeometry padding,
  }) {
    return CardState(
      width: width,
      height: height,
      backgroundColor: backgroundColor,
      borderRadius: borderRadius,
      padding: padding,
    );
  }

  static bool shouldHandleInteraction(VoidCallback? onTap) {
    return onTap != null;
  }
}
