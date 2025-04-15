import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

// 라우트
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/services/user_preferences_service.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_salary/budget_type_card.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_secure_info_viewmodel.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

// 위젯
import 'package:marshmellow/presentation/widgets/celebration/celebration.dart';

class SalaryCelebratePage extends ConsumerStatefulWidget {
  const SalaryCelebratePage({super.key});

  @override
  ConsumerState<SalaryCelebratePage> createState() => _SalaryCelebratePageState();
}

class _SalaryCelebratePageState extends ConsumerState<SalaryCelebratePage> {
  final _secureStorage = FlutterSecureStorage();
  String userName = '';
  bool showCelebration = true;
  bool showBudgetTypeOVerlay = false;

  @override
  void initState() {
    super.initState();

    _loadSHowCelebration();                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
  }

  Future<void> _loadSHowCelebration() async {
    try {
      final userSecureInfostate = ref.read(userSecureInfoProvider);
      final name = userSecureInfostate.userName;

      if (name != null) {
        setState(() {
          userName = name;
        });
      }

      print('🥕🥕User name: $userName');

      // 마운트되면 이후에 축하 표시
      WidgetsBinding.instance.addPostFrameCallback((_) {
        setState(() {
          showCelebration = true;
        });

        // 축하 5초 대기 후 카드 표시
        Future.delayed(const Duration(seconds: 3), () {
          if (mounted) {
            setState(() {
              showBudgetTypeOVerlay = true;

            });
          }
        });
      });
    } catch (e) {
      // 에러 처리
      print('Error loading user name: $e');
    }
  }

  void _navigateToBudgetTypePage() async {
    // 이 플로우 봣다고 체크하자!
    await UserPreferencesService.markBudgetFlowAsSeen();
    context.go(SignupRoutes.getBudgetTypeSelectionPath());
  }

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.of(context).size.height;
    final screenWidth = MediaQuery.of(context).size.width;

    return Scaffold(
      body: Stack(
      children: [

        // celebration 위젯이 아래로 가도록 순서 바꿈!
        if (showCelebration)
          CelebrationPopup(
            titleText: '야호!',
            subtitleText: 
              '${userName.isNotEmpty ? userName : '윤재은'} 님의\n월급날입니다!',
            characterImagePath: 'assets/images/characters/char_jump.png',
            confettiCount: 20,
            confettiDuration: 4000,
          ),

        // BudgetTypeCard를 오버레이로 표시
        if (showBudgetTypeOVerlay)
          Positioned.fill(
            child: Container(
              color: Colors.black.withOpacity(0.4),
              child: Center(
                child: AnimatedOpacity(
                  opacity: showBudgetTypeOVerlay ? 1.0 : 0.0,
                  duration: Duration(milliseconds: 500),
                  child: Container(
                    width: screenWidth * 0.85,
                    height: screenHeight * 0.65,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.2),
                          blurRadius: 15,
                          spreadRadius: 5,
                        ),
                      ],
                    ),
                    child: BudgetTypeCard(
                      onTapMoreDetails: _navigateToBudgetTypePage,
                    ),
                  ),
                ),
              ),
            ),
          ),
      ],
    )

    );
  }
}
