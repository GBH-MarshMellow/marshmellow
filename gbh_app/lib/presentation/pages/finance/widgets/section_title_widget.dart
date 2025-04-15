// presentation/pages/finance/widgets/section_title_widget.dart
import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class SectionTitleWidget extends StatelessWidget {
  final String title;
  
  const SectionTitleWidget({
    Key? key,
    required this.title,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 0.0),
      child: Text(
        title,
        style: AppTextStyles.bodySmall
      ),
    );
  }
}