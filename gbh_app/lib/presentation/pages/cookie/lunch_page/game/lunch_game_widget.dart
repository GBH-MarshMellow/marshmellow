import 'package:flame/game.dart';
import 'package:flutter/material.dart';
import 'lunch_game.dart';

class LunchGameWidget extends StatefulWidget {
  final List selectedMenus;
  final Function(List) onGameComplete;
  
  const LunchGameWidget({
    Key? key,
    required this.selectedMenus,
    required this.onGameComplete,
  }) : super(key: key);

  @override
  State<LunchGameWidget> createState() => LunchGameWidgetState();
}

class LunchGameWidgetState extends State<LunchGameWidget> {
  late LunchGame _game;
  BoundaryType _selectedBoundaryType = BoundaryType.DEFAULT;
  
  // 외부에서 접근할 수 있는 게임 제어 메서드들
  void startGame() {
    _game.startGame();
  }
  
  void resetGame() {
    _game.resetGame();
  }
  
  void changeBoundaryType(BoundaryType newType) {
    setState(() {
      _selectedBoundaryType = newType;
      _initializeGame();
    });
  }

  @override
  void initState() {
    super.initState();
    _initializeGame();
  }

  void _initializeGame() {
    _game = LunchGame(
      selectedMenus: widget.selectedMenus,
      boundaryType: _selectedBoundaryType,
    );
    
    _game.onGameComplete = widget.onGameComplete;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // 게임 영역
        Expanded(
          child: GameWidget(game: _game),
        ),
      ],
    );
  }
}