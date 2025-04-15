import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/theme/app_colors.dart';

// 라우트
import 'package:go_router/go_router.dart';
import 'package:marshmellow/router/routes/cookie_routes.dart';

// 위젯
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/celebration/celebration.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/quit/quit_sequence.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/quit/quit_failure.dart';

// ViewModel
import 'package:marshmellow/presentation/viewmodels/quit/quit_viewmodel.dart';

class QuitPage extends ConsumerStatefulWidget {
  const QuitPage({super.key});

  @override
  ConsumerState<QuitPage> createState() => _QuitPageState();
}

class _QuitPageState extends ConsumerState<QuitPage> {
  final _numberFormat = NumberFormat('#,###', 'ko_KR');
  bool _dataLoaded = false;
  bool _surviveThreeMonths = false;

  @override
  void initState() {
    super.initState();

    // 화면이 완전히 빌드된 후 데이터 로드
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(quitViewModelProvider.notifier).loadAllData();
    });
  }

  // 3개월 버틸 수 있는지 확인하는 메서드
  bool canSurviveThreeMonths(int availableAmount, int averageSpending) {
    int remainingAmount = availableAmount;

    // 3개월 동안 감소를 시뮬레이션
    for (int i = 0; i < 3; i++) {
      remainingAmount -= averageSpending;
      if (remainingAmount < 0) {
        return false; // 3개월 내에 바닥남
      }
    }

    return true; // 3개월 이상 버틸 수 있음
  }

  @override
  Widget build(BuildContext context) {
    final quitState = ref.watch(quitViewModelProvider);

    // 로딩 중일 때는 로딩 인디케이터만 표시
    if (quitState.isLoading) {
      return Scaffold(
        appBar: CustomAppbar(
          title: '퇴사 망상',
          actions: [
            IconButton(
              onPressed: () => context.push(CookieRoutes.getQuitInfoPath()),
              icon: SvgPicture.asset(IconPath.question),
            ),
          ],
        ),
        body: Center(
          child:           
            Lottie.asset(
            'assets/images/loading/loading_simple.json',
            width: 140,  // 원하는 크기로 조정
            height: 140,
            fit: BoxFit.contain,
            ),
        ),
      );
    }

    // 데이터 로드 후 3개월 생존 여부 확인
    if (quitState.availableAmount != null &&
        quitState.averageSpending != null &&
        !_dataLoaded) {
      _surviveThreeMonths = canSurviveThreeMonths(
        quitState.availableAmount!.availableAmount,
        quitState.averageSpending!.averageMonthlySpending,
      );

      _dataLoaded = true;

      // 3개월 이상 버틸 수 있을 때만 축하 효과 표시
      if (_surviveThreeMonths) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          showRetirementCelebration(context);
        });
      }
    }

    return Scaffold(
      appBar: CustomAppbar(
        title: '퇴사 망상',
        actions: [
          IconButton(
            onPressed: () => context.push(CookieRoutes.getQuitInfoPath()),
            icon: SvgPicture.asset(IconPath.question),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: EdgeInsets.symmetric(
            horizontal: MediaQuery.of(context).size.width * 0.04,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 20),
              Text('한달 평균 지출', style: AppTextStyles.bodyMedium),
              const SizedBox(height: 10),

              // averageSpending이 null이 아닐 때만 표시
              if (quitState.averageSpending != null)
                Row(
                  children: [
                    Text(
                      _numberFormat.format(
                          quitState.averageSpending!.averageMonthlySpending),
                      style: AppTextStyles.modalMoneyTitle,
                    ),
                    const SizedBox(width: 5),
                    Text('원', style: AppTextStyles.bodyMedium),
                  ],
                ),

              const SizedBox(height: 50),

              // 3개월 생존 가능 여부에 따라 다른 위젯 표시
              _surviveThreeMonths
                  ? const QuitSequence() // 3개월 이상 버틸 수 있으면 시퀀스 표시
                  : const QuitFailure(), // 3개월 내에 바닥나면 경고 표시

              const SizedBox(height: 30),
            ],
          ),
        ),
      ),
    );
  }
}
