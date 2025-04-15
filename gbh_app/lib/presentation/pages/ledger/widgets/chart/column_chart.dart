import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';
import 'package:intl/intl.dart';

import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

// 차트 데이터 모델
class ChartData {
  final String label;
  final double value;
  final Color color;

  ChartData({
    required this.label,
    required this.value,
    this.color = AppColors.pinkPrimary,
  });
}

class ColumnChart extends StatelessWidget {
  final List<ChartData> data;

  const ColumnChart({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    return SfCartesianChart(
      primaryXAxis: CategoryAxis(
        majorGridLines: const MajorGridLines(width: 0),
        labelStyle: AppTextStyles.bodySmall,
      ),
      primaryYAxis: NumericAxis(
        isVisible: false,
        majorGridLines: const MajorGridLines(width: 0),
        axisLine: const AxisLine(width: 0),
        labelFormat: '{value}',
        labelStyle: AppTextStyles.periodSmall.copyWith(
          fontSize: 8,
          fontWeight: FontWeight.w300,
        ),
      ),
      tooltipBehavior: TooltipBehavior(
        enable: true,
        header: '',
        format: 'point.y',
        builder: (dynamic data, dynamic point, dynamic series, int pointIndex,
            int seriesIndex) {
          final value = point.y as double;
          return Container(
            padding: const EdgeInsets.all(8),
            child: Text(
              '${NumberFormat('#,###').format(value)}원',
              style: AppTextStyles.bodySmall.copyWith(
                color: AppColors.whiteLight,
              ),
            ),
          );
        },
      ),
      series: <CartesianSeries>[
        ColumnSeries<ChartData, String>(
          // 애니메이션 효과
          animationDuration: 1500,
          dataSource: data,
          xValueMapper: (ChartData data, _) => data.label,
          yValueMapper: (ChartData data, _) => data.value,
          pointColorMapper: (ChartData data, _) => data.color,
          borderRadius: BorderRadius.circular(5),
          width: 0.7, // 컬럼 너비
          spacing: 0.2, // 컬럼 간격
          dataLabelSettings: DataLabelSettings(
            isVisible: false,
            labelAlignment: ChartDataLabelAlignment.top,
            textStyle: AppTextStyles.moneyGraphMedium.copyWith(fontSize: 10),
          ),
        ),
      ],
      plotAreaBorderWidth: 0,
      margin: const EdgeInsets.all(16),
      borderColor: Colors.transparent,
      backgroundColor: Colors.transparent,
    );
  }
}
