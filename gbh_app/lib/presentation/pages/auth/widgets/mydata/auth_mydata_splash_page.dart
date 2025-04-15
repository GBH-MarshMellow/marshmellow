import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/di/providers/core_providers.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/mydata/auth_mydata_cert_select_modal.dart';
import 'package:marshmellow/presentation/viewmodels/auth/certificate_notifier.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

/*
  mm인증서 스플래쉬 UI
*/
class AuthMydataSplashPage extends ConsumerStatefulWidget {
  const AuthMydataSplashPage({Key? key}) : super(key: key);

  @override
  ConsumerState<AuthMydataSplashPage> createState() => _AuthMydataSplashPageState();
}

class _AuthMydataSplashPageState extends ConsumerState<AuthMydataSplashPage> 
    with TickerProviderStateMixin {

  late AnimationController _firstRowController;
  late AnimationController _secondRowController;
  late AnimationController _thirdRowController;

  late Animation<Offset> _firstRowAnimation;
  late Animation<Offset> _secondRowAnimation;
  late Animation<Offset> _thirdRowAnimation;

  // 버튼 활성화 상태
  bool _isButtonEnabled = false;
  bool _isLoading = false;

  // 은행 아이콘을 불러오자
  final List<String> bankIconPaths = [
    IconPath.kbBank,
    IconPath.shinhanBank,
    IconPath.wooriBank,
    IconPath.nhBank,
    IconPath.ibkBank,
    IconPath.kakaoBank,
    // IconPath.ssafyBank,
    IconPath.dgBank,
    IconPath.gjBank,
    IconPath.citiBank,
    IconPath.mgBank,
    IconPath.hanaBank,
    IconPath.scBank,
    // IconPath.koreaBank,
    IconPath.jejuBank,
    // IconPath.jbBank,
  ];

  String _userName = '사용자'; // 기본값

  @override
  void initState() {
    super.initState();

    // 애니메이션 컨트롤러 초기화
    _firstRowController = AnimationController(
      duration: const Duration(seconds: 10),
      vsync: this,
    )..repeat(reverse: true);

    // 좌상단에서 우하단으로 이동하는 슬라이딩 애니메이션 정의
    _firstRowAnimation = Tween<Offset>(
      begin: const Offset(0, 0),
      end: const Offset(-1, 0),
    ).animate(CurvedAnimation(
      parent: _firstRowController, 
      curve: Curves.linear,
    ));

    _secondRowController = AnimationController(
      duration: const Duration(seconds: 12),
      vsync: this,
    )..repeat(reverse: true);

    _secondRowAnimation = Tween<Offset>(
      begin: const Offset(-1, 0),
      end: const Offset(0, 0),
    ).animate(CurvedAnimation(
      parent: _secondRowController, 
      curve: Curves.linear,
    ));

    _thirdRowController = AnimationController(
      duration: const Duration(seconds: 8),
      vsync: this,
    )..repeat(reverse: true);

    _thirdRowAnimation = Tween<Offset>(
      begin: const Offset(0, 0),
      end: const Offset(-1, 0),
    ).animate(CurvedAnimation(
      parent: _thirdRowController, 
      curve: Curves.linear,
    ));

    // 사용자 이름 불러오기
    _loadUserName();

    // 페이지 진입시 인증서 상태 확인
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _autoCheckCertificatePemStatus();
    });
  }

  @override
  void dispose() {
    _firstRowController.dispose();
    _secondRowController.dispose();
    _thirdRowController.dispose();
    super.dispose();
  }

  // 사용자 이름 불러오는 함수
  Future<void> _loadUserName() async {
    final secureStorage = ref.read(secureStorageProvider);
    final name = await secureStorage.read(key: StorageKeys.userName);
    if (name != null && mounted) {
      setState(() {
        _userName = name;
      });
    }
  }

  Future<void> _autoCheckCertificatePemStatus() async {
    setState(() {
      _isLoading = true;
    });

    try {
      // 인증서상태 확인 api호출
      await ref.read(certificateProvider.notifier).checkCertificateStatus();

      // 위젯마운팅확인
      if (!mounted) return;

      final certState = ref.read(certificateProvider);

      // 로딩 상태 업데이트
      setState(() {
        _isLoading = false;
      });

      if (certState.hasCertificate) {
        context.showAuthMydataCertSelect(_userName);
      } else {
        // context.go(SignupRoutes.getMyDataEmailPath());
      }
    } catch (e) {
      if (!mounted) return;

      // 오류 처리
      setState(() {
        _isLoading = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('인증서 확인 중 오류가 발생했습니다: $e'))
      );
    }
  }



  // 사용자가 버튼을 누른 경우 호출 메서드 : 인증서 상태 확인 및 처리
  Future<void> _checkCertificateStatus() async {
    setState(() {
      _isLoading = true;
    });

    try {
      // 인증서 상태 확인 API 호출
      await ref.read(certificateProvider.notifier).checkCertificateStatus();

      // 위젯 마운팅 확인
      if (!mounted) return;

      final certState = ref.read(certificateProvider);

      // UI 업데이트
      setState(() {
        _isLoading = false;
      });

      // 인증서 상태에 따른 분기 처리
      if (certState.hasCertificate) {
        // 인증서 존재 - 인증서 선택 모달 표시
        context.showAuthMydataCertSelect(_userName);
      } else {
        // 인증서 존재X - 이메일 입력 페이지(인증서생성시작) 이동
        context.go(SignupRoutes.getMyDataEmailPath());
      }
    } catch (e) {
      if (!mounted) return;

      // 오류 처리
      setState(() {
        _isLoading = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('인증서 확인 중 오류가 발생했습니다: $e'))
      );
    }
  }

  // 체크박스 상태 변경 처리
  void _toggleButtonEnabled() {
    setState(() {
      _isButtonEnabled = !_isButtonEnabled;
    });
  }

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.of(context).size.height;
    final screenWidth = MediaQuery.of(context).size.width;

    const double iconSize = 75;

    final List<Widget> firstRowIcons = _createDuplicatedIconRow(bankIconPaths.sublist(0, 4), iconSize);
    final List<Widget> secondRowIcons = _createDuplicatedIconRow(bankIconPaths.sublist(4, 8), iconSize);
    final List<Widget> thirdRowIcons = _createDuplicatedIconRow(bankIconPaths.sublist(8, bankIconPaths.length), iconSize);

    return Scaffold(
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                const SizedBox(height: 50),
                const SizedBox(height: 50),
                Text('$_userName 님의', style: AppTextStyles.mainTitle),
                const Text('자산을 한 번에 찾아보세요', style: AppTextStyles.mainTitle),
                const SizedBox(height: 10),
                Text('단 30초면 모든 기관을 찾고 연결할 수 있어요', 
                     style: AppTextStyles.bodySmall.copyWith(color: AppColors.disabled)),

                // 은행 아이콘 그리드
                Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      _buildSlidingRow(firstRowIcons, _firstRowAnimation, screenWidth),
                      const SizedBox(height: 1),
                      _buildSlidingRow(secondRowIcons, _secondRowAnimation, screenWidth),
                      const SizedBox(height: 1),
                      _buildSlidingRow(thirdRowIcons, _thirdRowAnimation, screenWidth),
                    ],
                  )
                ),

                Center(
                  child: InkWell(
                    onTap: _toggleButtonEnabled,
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          padding: const EdgeInsets.all(4),
                          decoration: BoxDecoration(
                            color: _isButtonEnabled ? AppColors.backgroundBlack : Colors.transparent,
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: _isButtonEnabled ? AppColors.backgroundBlack : AppColors.disabled,
                            ),
                          ),
                          child: Icon(
                            Icons.check,
                            color: _isButtonEnabled ? AppColors.whiteLight : Colors.transparent,
                            size: 14,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text('17개 금융사 선택', 
                             style: AppTextStyles.bodySmall.copyWith(color: AppColors.disabled), 
                             textAlign: TextAlign.center,)
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 16),

                Button(
                  text: '찾아보기',
                  width: screenWidth * 0.9,
                  height: 60,
                  onPressed: _isButtonEnabled ? _checkCertificateStatus : null,
                  isDisabled: !_isButtonEnabled,
                ),

                SizedBox(height: screenHeight * 0.05),
              ],
            ),
          ),
          
          // 로딩 인디케이터
          if (_isLoading)
            CustomLoadingIndicator(text: '인증서 정보를 확인중입니다', backgroundColor: AppColors.whiteLight, opacity: 0.9,)
        ],
      ),
    );
  }

  List<Widget> _createDuplicatedIconRow(List<String> icons, double size) {
    List<Widget> iconWidgets = [];

    for (int i = 0; i < 2; i++) {
      for (String path in icons) {
        iconWidgets.add(
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 5),
            child: SvgPicture.asset(
              path,
              width: size,
              height: size,
            ),
          ),
        );
      }
    }
    return iconWidgets;
  }

  Widget _buildSlidingRow(List<Widget> icons, Animation<Offset> animation, double screenWidth) {
    return SizedBox(
      height: 100,
      width: screenWidth,
      child: ClipRect(
        child: SlideTransition(
        position: animation,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: icons,
        ),
      ),)
    );
  }
}
