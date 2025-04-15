import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/services/user_preferences_service.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart'; // 텍스트 스타일 import 추가
import 'package:marshmellow/core/theme/app_colors.dart'; // 테마 import 추가
import 'package:marshmellow/core/config/app_config.dart';
import 'package:marshmellow/core/utils/back_gesture/controller.dart';
import 'package:marshmellow/core/utils/back_gesture/detector.dart';
import 'package:marshmellow/data/models/my/user_detail_info.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_info_viewmodel.dart';
import 'package:marshmellow/router/app_router.dart'; // 라우터 import
import 'package:marshmellow/di/providers/lifecycle_provider.dart';
import 'package:marshmellow/presentation/widgets/datepicker/date_picker_overlay.dart';
import 'package:marshmellow/di/providers/calendar_providers.dart';
import 'package:marshmellow/presentation/viewmodels/budget/budget_viewmodel.dart'; // 예산 뷰모델 추가

// Flutter 로컬라이제이션 패키지 추가
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';
import 'package:marshmellow/router/routes/budget_routes.dart';
import 'package:marshmellow/core/utils/widgets/widget_service.dart';

// LifecycleEventHandler 클래스 구현
class LifecycleEventHandler extends WidgetsBindingObserver {
  final Future<void> Function() resumeCallBack;
  final Future<void> Function() suspendingCallBack;

  LifecycleEventHandler({
    required this.resumeCallBack,
    required this.suspendingCallBack,
  });

  @override
  Future<void> didChangeAppLifecycleState(AppLifecycleState state) async {
    switch (state) {
      case AppLifecycleState.resumed:
        await resumeCallBack();
        break;
      case AppLifecycleState.inactive:
      case AppLifecycleState.paused:
      case AppLifecycleState.detached:
        await suspendingCallBack();
        break;
      default:
        break;
    }
  }
}

// 공유된 URL을 저장할 전역 상태 제공자
final sharedUrlProvider = StateProvider<String?>((ref) => null);

// 메서드 채널 설정
const MethodChannel _channel = MethodChannel('app.channel.shared.data');

class App extends ConsumerStatefulWidget {
  final GoRouter router;
  final BackGestureController backGestureController;

  const App(
      {super.key, required this.router, required this.backGestureController});

  @override
  ConsumerState<App> createState() => _AppState();
}

class _AppState extends ConsumerState<App> {
  @override
  void initState() {
    super.initState();

    // 메서드 채널 리스너 설정
    _setupMethodChannelListener();

    // 초기 공유 텍스트 확인
    _getInitialSharedText();

    // 라이프사이클 매니저 초기화 - 프로바이더를 읽는 것만으로 초기화됨
    ref.read(appLifecycleManagerProvider);
    ref.read(paydayFetchProvider);

    // ✨ 위젯 업데이트 리스너 설정
    _setupWidgetUpdateListener();
  }

  // ✨ 위젯 업데이트 리스너 설정
  void _setupWidgetUpdateListener() {
    // 앱 라이프사이클 상태 변화 감지
    WidgetsBinding.instance.addObserver(LifecycleEventHandler(
      resumeCallBack: () async {
        // 앱이 포그라운드로 돌아왔을 때 위젯 데이터 새로고침
        try {
          final budgetProviderState = ref.read(budgetProvider.notifier);
          await budgetProviderState.fetchBudgets();
        } catch (e) {
          print('위젯 업데이트 오류: $e');
        }
      },
      suspendingCallBack: () async {
        // 앱이 백그라운드로 갈 때 마지막으로 위젯 업데이트
        try {
          final state = ref.read(budgetProvider);
          if (state.dailyBudget != null) {
            await WidgetService.updateBudgetWidget(
                state.dailyBudget!.dailyBudgetAmount);
          }
        } catch (e) {
          print('백그라운드 위젯 업데이트 오류: $e');
        }
      },
    ));
  }

  // 위시 크롤링 자동생성 : 메서드 채널 리스너 설정
  void _setupMethodChannelListener() {
    _channel.setMethodCallHandler((MethodCall call) async {
      if (call.method == 'sharedText') {
        final String? sharedText = call.arguments as String?;
        if (sharedText != null && sharedText.isNotEmpty) {
          // 공유된 URL을 상태에 저장
          ref.read(sharedUrlProvider.notifier).state = sharedText;

          // 위시리스트 생성 페이지로 이동
          widget.router.go(BudgetRoutes.getWishlistCreatePath());
        }
      }
      return null;
    });
  }

  // 위시 크롤링 자동생성 : 초기 공유 텍스트 확인
  Future<void> _getInitialSharedText() async {
    try {
      final String? sharedText = await _channel.invokeMethod('getSharedText');
      if (sharedText != null && sharedText.isNotEmpty) {
        // 공유된 URL을 상태에 저장
        ref.read(sharedUrlProvider.notifier).state = sharedText;

        // 위시리스트 생성 페이지로 이동
        Future.microtask(() {
          widget.router.go(BudgetRoutes.getWishlistCreatePath());
        });
      }
    } catch (e) {
      print('초기 공유 텍스트 가져오기 오류: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
        title: 'MMApp',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          primaryColor: AppColors.background,
          scaffoldBackgroundColor: AppColors.background,
          fontFamily: AppTextStyles.fontFamily,
          textTheme: TextTheme(
            bodyLarge: AppTextStyles.bodyLarge,
            titleLarge: AppTextStyles.appBar,
          ),
          colorScheme: ColorScheme.light(
            primary: AppColors.textPrimary,
          ),
          useMaterial3: true,
        ),
        // 한국어 로케일 설정
        locale: const Locale('ko', 'KR'),
        // 지원하는 로케일 목록
        supportedLocales: const [
          Locale('ko', 'KR'), // 한국어
          Locale('en', 'US'), // 영어 (필요한 경우)
        ],
        // 로컬라이제이션 대리자
        localizationsDelegates: const [
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        routerConfig: widget.router,
        // 스와이프 뒤로가기 제스처 감지 추가
        builder: (context, child) {
          return DatePickerOverlay(
            child: SwipeBackDetector(
              controller: widget.backGestureController,
              router: widget.router,
              child: child!,
            ),
          );
        });
  }
}
