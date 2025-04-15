import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

class NoRecentSearches extends StatelessWidget {
  const NoRecentSearches({super.key});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        children: [
          const SizedBox(height: 30),
          Image.asset(
            'assets/images/characters/char_chair_phone.png',
            height: 150,
          ),
          const SizedBox(height: 30),
          Text(
            '수입 지출 내역 또는 메모 내용으로\n손쉽게 검색해보세요',
            style:
                AppTextStyles.bodyMedium.copyWith(fontWeight: FontWeight.w400),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
