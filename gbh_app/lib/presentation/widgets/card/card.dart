import 'package:flutter/material.dart';
import 'package:marshmellow/presentation/widgets/card/card_logic.dart';

class CustomCard extends StatelessWidget {
  final Widget child;
  final double? width;
  final double? height;
  final Color backgroundColor;
  final double borderRadius;
  final EdgeInsetsGeometry padding;
  final VoidCallback? onTap;

  const CustomCard({
    Key? key,
    required this.child,
    this.width,
    this.height,
    this.backgroundColor = Colors.white,
    this.borderRadius = 10.0,
    this.padding = const EdgeInsets.all(16.0),
    this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final cardState = CardLogic.getCardState(
      width: width,
      height: height,
      backgroundColor: backgroundColor,
      borderRadius: borderRadius,
      padding: padding,
    );

    return GestureDetector(
      onTap: CardLogic.shouldHandleInteraction(onTap) ? onTap : null,
      child: Container(
        width: cardState.width,
        height: cardState.height,
        padding: cardState.padding,
        decoration: BoxDecoration(
          color: cardState.backgroundColor,
          borderRadius: BorderRadius.circular(cardState.borderRadius),
        ),
        child: child,
      ),
    );
  }
}


// 사용법 예시
/*


// 1. 크기 지정
CustomCard(
  width: 200,
  height: 150,
  child: Text('크기가 지정된 카드'),
)

// 2. 복합 컨텐츠 
CustomCard(
  backgroundColor: AppColors.bluePrimary,
  child: Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text('제목', style: AppTextStyles.bodySmall),
      SizedBox(height: 8),
      Text(
        '여기에 카드 내용을 적습니다. 여러 줄의 텍스트와 다양한 위젯을 포함할 수 있습니다.',
        style: AppTextStyles.bodyMedium.copyWith(fontWeight: FontWeight.w300),
      ),
    ],
  ),
  onTap: () {
    print('카드가 탭됨');
  },
)

// 3. 여러 카드를 가로로 배치
Row(
  mainAxisAlignment: MainAxisAlignment.center,
  children: [
    CustomCard(
      width: MediaQuery.of(context).size.width * 0.45,
      height: MediaQuery.of(context).size.height * 0.13,
      backgroundColor: AppColors.pinkPrimary,
      onTap: () {
        print('카드가 탭됨');
      },
      child: const Text('카드 내용'),
    ),
    SizedBox(width: MediaQuery.of(context).size.width * 0.01),
    CustomCard(
      width: MediaQuery.of(context).size.width * 0.45,
      height: MediaQuery.of(context).size.height * 0.13,
      backgroundColor: AppColors.bluePrimary,
      onTap: () {
        print('카드가 탭됨');
      },
      child: const Text('카드 내용'),
    ),
  ],
)

*/