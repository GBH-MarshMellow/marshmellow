import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/card/card.dart';

class FinanceCard extends StatelessWidget {
  final String title;
  final int amount;
  final Color backgroundColor;
  final VoidCallback? onTap;
  final double? width;
  final double? height;
  final double? borderRadius;

  const FinanceCard({
    Key? key,
    required this.title,
    required this.amount,
    required this.backgroundColor,
    this.onTap,
    this.width,
    this.height,
    this.borderRadius,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return CustomCard(
      width: width ?? MediaQuery.of(context).size.width * 0.43,
      height: height ?? MediaQuery.of(context).size.height * 0.11,
      borderRadius: borderRadius ?? 5,
      backgroundColor: backgroundColor,
      onTap: onTap,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(title, style: AppTextStyles.bodySmall),
          const SizedBox(height: 8),
          Row(
            children: [
              Text(
                  '${amount.toString().replaceAllMapped(RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'), (Match m) => '${m[1]},')}',
                  style: AppTextStyles.bodyLarge
                      .copyWith(fontWeight: FontWeight.w600)),
              const SizedBox(width: 4),
              Text('원', style: AppTextStyles.bodySmall),
            ],
          )
        ],
      ),
    );
  }
}
