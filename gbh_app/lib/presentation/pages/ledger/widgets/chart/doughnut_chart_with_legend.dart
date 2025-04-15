import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/chart/category_bottom_sheet_modal.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';

class ChartData {
  final String title;
  final double value; // %값
  final double? amount; // 금액
  final Color color;
  final List<Transaction>? transactions;

  ChartData({
    required this.title,
    required this.value,
    this.amount,
    required this.color,
    this.transactions,
  });
}

class DoughnutChartWithLegend extends StatelessWidget {
  final List<ChartData> data;
  final double? height;
  final bool isLoading;
  final List<Transaction>? transactions;
  final WidgetRef ref;

  const DoughnutChartWithLegend({
    super.key,
    required this.data,
    this.height,
    this.isLoading = false,
    this.transactions,
    required this.ref,
  });

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.of(context).size.height;
    final chartHeight = height ?? screenHeight * 0.35;

    // 로딩 중인 경우 로딩 인디케이터 표시
    if (isLoading) {
      return SizedBox(
        height: chartHeight,
        child: Center(
          child: CircularProgressIndicator(
            color: AppColors.textPrimary,
          ),
        ),
      );
    }

    // 데이터가 없는 경우 처리
    if (data.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 40),
          child: Column(
            children: [
              Image.asset(
                'assets/images/characters/char_melong.png',
                width: 150,
                height: 150,
              ),
              const SizedBox(height: 16),
              Text(
                '해당 기간에 지출 내역이 없습니다.',
                style: AppTextStyles.bodyMedium.copyWith(
                  color: AppColors.textSecondary,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return Column(
      children: [
        // 도넛 차트
        SizedBox(
          height: chartHeight,
          child: DoughnutChart(data: data),
        ),
        const SizedBox(height: 20),

        // 차트 범례
        SizedBox(
          height: data.length * 40, // 항목 개수에 따라 동적 높이 설정
          child: ListView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: data.length,
            itemBuilder: (context, index) {
              final item = data[index];
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 4.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // 왼쪽 영역: 원형 마커와 카테고리 이름
                    Row(
                      children: [
                        Container(
                          width: 12,
                          height: 12,
                          decoration: BoxDecoration(
                            color: item.color,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(item.title,
                            style: AppTextStyles.bodyMedium
                                .copyWith(fontSize: 14)),
                        const SizedBox(width: 15),
                        Text(
                          '${item.value.toStringAsFixed(1)}%',
                          style: AppTextStyles.bodyMedium.copyWith(
                            fontSize: 14,
                            color: AppColors.greyPrimary,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(width: 10),
                    // 금액 표시 추가
                    Row(
                      children: [
                        if (item.amount != null) ...[
                          const SizedBox(width: 10),
                          Text(
                            '${NumberFormat('#,###').format(item.amount!.toInt())}원',
                            style: AppTextStyles.bodyMedium.copyWith(
                              fontSize: 14,
                              color: AppColors.textPrimary,
                            ),
                          ),
                        ],
                        const SizedBox(width: 10),
                        // 오른쪽 영역: 화살표 아이콘
                        SizedBox(
                          width: 14,
                          height: 30,
                          child: IconButton(
                            padding: EdgeInsets.zero,
                            constraints: const BoxConstraints(),
                            onPressed: () {
                              // item.transactions를 직접 사용
                              showCategoryBottomSheetModal(
                                context: context,
                                ref: ref,
                                categoryName: item.title,
                                transactions: item.transactions ?? [],
                              );
                            },
                            icon: SvgPicture.asset(
                              IconPath.caretRight,
                              width: 14,
                              height: 14,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}

// 기존 도넛 차트 컴포넌트
class DoughnutChart extends StatefulWidget {
  final List<ChartData> data;

  const DoughnutChart({super.key, required this.data});

  @override
  State<DoughnutChart> createState() => _DoughnutChartState();
}

class _DoughnutChartState extends State<DoughnutChart>
    with SingleTickerProviderStateMixin {
  int? touchedIndex;
  late AnimationController _animationController;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500), // 애니메이션 속도 줄이기
    );

    _animation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOut, // 좀 더 자연스러운 애니메이션
    );

    _animationController.forward();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Opacity(
          opacity: _animation.value > 0.1 ? 1.0 : 0.0, // 초반 깜빡임 방지
          child: PieChart(
            PieChartData(
              sectionsSpace: 2, // 구분선 추가
              centerSpaceRadius: 50, // 도넛 스타일
              sections: _generateSections(_animation.value),
              borderData: FlBorderData(show: false),
              pieTouchData: PieTouchData(
                touchCallback: (FlTouchEvent event, pieTouchResponse) {
                  setState(() {
                    if (!event.isInterestedForInteractions ||
                        pieTouchResponse == null ||
                        pieTouchResponse.touchedSection == null) {
                      touchedIndex = -1;
                      return;
                    }
                    touchedIndex =
                        pieTouchResponse.touchedSection!.touchedSectionIndex;
                  });
                },
              ),
            ),
          ),
        );
      },
    );
  }

  List<PieChartSectionData> _generateSections(double animationValue) {
    return widget.data.asMap().entries.map((entry) {
      final index = entry.key;
      final item = entry.value;
      final baseRadius = 80.0;
      final isTouched = touchedIndex == index;
      final radius = isTouched ? baseRadius * 1.2 : baseRadius;
      final animatedValue = item.value * animationValue;
      final showTitle = animationValue > 0.8;
      final fontSize = isTouched ? 16.0 : 12.0;
      final fontWeight = isTouched ? FontWeight.bold : FontWeight.normal;

      // 색상의 밝기 계산 (어두운 색상이면 흰색 텍스트, 밝은 색상이면 검정색 텍스트)
      final brightness = ThemeData.estimateBrightnessForColor(item.color);
      final textColor = brightness == Brightness.dark
          ? AppColors.whiteDark
          : AppColors.textPrimary;

      return PieChartSectionData(
        color: item.color,
        value: animatedValue,
        title: showTitle
            ? '${item.title}\n${animatedValue.toStringAsFixed(1)}%'
            : '',
        radius: radius,
        titleStyle: AppTextStyles.bodyMedium.copyWith(
          color: textColor,
          fontSize: fontSize,
          fontWeight: fontWeight,
        ),
      );
    }).toList();
  }
}
