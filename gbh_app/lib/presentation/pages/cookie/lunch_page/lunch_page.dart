import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/constants/lunch_menu_data.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/TopTriangleBubbleWidget.dart';
import 'package:marshmellow/presentation/viewmodels/lunch/lunch_view_model.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/router/routes/cookie_routes.dart';
import 'package:path/path.dart';

class LunchPage extends ConsumerWidget {
  const LunchPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // 뷰모델 가져오기
    final lunchViewModel = ref.watch(lunchViewModelProvider);
    
    return Scaffold(
      appBar: CustomAppbar(
        title: '점심 메뉴 추천',
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // 상단 타이틀
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                Container(
                  width: 200,
                  child: Text(
                    lunchViewModel.selectedMenus.isNotEmpty 
                        ? lunchViewModel.selectedMenus.last.name
                        : '메뉴를 선택해주세요',
                    style: TextStyle(
                      color: AppColors.disabled,
                      fontSize: 20,
                      fontWeight: FontWeight.w400,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ],
            ),
            // const SizedBox(height: 24),
            
            // 선택된 메뉴 표시 영역
            _buildSelectedMenusSection(lunchViewModel),
            const SizedBox(height: 24),
            
            // 메뉴 그리드 영역
            Expanded(
              child: _buildMenuGrid(context, lunchViewModel),
            ),
            
            // 시작 버튼
            Button(
              text: '시작',
              onPressed: () {
                // 시작 페이지로 이동
                context.replace(CookieRoutes.getLunchRunPath());
              },
            ),
            const SizedBox(height: 30),
          ],
        ),
      ),
    );
  }

  // 선택된 메뉴 목록을 보여주는 위젯 (2행 5열 그리드)
  Widget _buildSelectedMenusSection(LunchViewModel viewModel) {
    // 2행 4열 = 총 8개의 슬롯
    const int rows = 2;
    const int columns = 4;
    const int totalSlots = rows * columns; // 8개 슬롯
    
    return Container(
      height: 140, // 그리드 높이 조정
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // 그리드 설명 텍스트
          Container(
            // color: Colors.amber,
            width: 140,
            height: 150,
            child: Stack(
              alignment: Alignment.center,
              children: [
                Positioned(
                  right: 10,
                  bottom: 30,
                  child: Container(
                    width: 120,
                    height: 120,
                    alignment: Alignment.centerLeft,
                    child: Image.asset(
                      'assets/images/characters/char_hat.png',
                      fit: BoxFit.contain,
                    ),
                  ),
                ),

                Positioned(
                  left: 30,
                  top: 33,
                  child: Transform.scale(
                    scale: 4,
                    child: Lottie.asset(
                      'assets/images/loading/mypot.json',
                      width: 60,
                      height: 60,
                      fit: BoxFit.contain,
                    ),
                  ),
                ),
              ],
            ),
          ),
          // 그리드 영역
          Expanded(
            child: Align(
              alignment: Alignment.bottomCenter, // 하단 중앙 정렬
              child: GridView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(), // 스크롤 비활성화
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: columns, // 4열
                  crossAxisSpacing: 8,
                  mainAxisSpacing: 8,
                  childAspectRatio: 1, // 정사각형
                ),
                itemCount: totalSlots, // 총 8개 슬롯
                itemBuilder: (context, index) {
                  // 선택된 메뉴가 있는 경우 메뉴 표시, 없는 경우 빈 슬롯 표시
                  if (index < viewModel.selectedMenus.length) {
                    return _buildFilledMenuSlot(viewModel, index);
                  } else {
                    return _buildEmptyMenuSlot();
                  }
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  // 선택된 메뉴가 채워진 슬롯 위젯
  Widget _buildFilledMenuSlot(LunchViewModel viewModel, int index) {
    final menu = viewModel.selectedMenus[index];
    
    return GestureDetector(
      onTap: () {
        // 선택 취소
        viewModel.unselectMenuAt(index);
      },
      child: Container(
        decoration: BoxDecoration(
          color: AppColors.blackPrimary,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: AppColors.buttonBlack),
        ),
        child: Stack(
          children: [
            // 메뉴 이미지
            Center(
              child: Image.asset(
                menu.imagePath,
                width: 40,
                height: 40,
              ),
            ),
            // 삭제 아이콘
            Positioned(
              top: 2,
              right: 2,
              child: Container(
                decoration: const BoxDecoration(
                  color: Color.fromARGB(255, 0, 0, 0),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.close, size: 14, color: Color.fromARGB(255, 122, 122, 122)),
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  // 빈 메뉴 슬롯 위젯
  Widget _buildEmptyMenuSlot() {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.divider,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppColors.blackLight),
      ),
    );
  }

  // 메뉴 그리드 위젯
  Widget _buildMenuGrid(BuildContext context, LunchViewModel viewModel) {
    return GridView.builder(
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 4,  // 4x4 그리드
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 1,  // 정사각형 버튼
      ),
      itemCount: allLunchMenus.length,
      itemBuilder: (context, index) {
        final menu = allLunchMenus[index];
        return _buildMenuButton(context, menu, viewModel);
      },
    );
  }

// 개별 메뉴 버튼 위젯
Widget _buildMenuButton(
  BuildContext context,
  LunchMenu menu,
  LunchViewModel viewModel,
) {
  return _AnimatedMenuButton(
    menu: menu,
    viewModel: viewModel,
    onTap: () {
      if (!viewModel.isMaxSelected) {
        viewModel.selectMenu(menu);
      } else {
        // 최대 선택 개수에 도달하면 알림 표시
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('최대 8개까지 선택 가능합니다'),
            duration: Duration(seconds: 1),
          ),
        );
      }
    },
  );
}
}

// 애니메이션이 적용된 메뉴 버튼
class _AnimatedMenuButton extends StatefulWidget {
  final LunchMenu menu;
  final LunchViewModel viewModel;
  final VoidCallback onTap;

  const _AnimatedMenuButton({
    required this.menu,
    required this.viewModel,
    required this.onTap,
  });

  @override
  State<_AnimatedMenuButton> createState() => _AnimatedMenuButtonState();
}

class _AnimatedMenuButtonState extends State<_AnimatedMenuButton> 
    with SingleTickerProviderStateMixin {
  bool _isPressed = false;
  
  // 애니메이션 완료 여부를 추적하기 위한 변수
  bool _animationComplete = true;
  
  // 탭 애니메이션 실행 함수
  void _playTapAnimation() {
    if (_animationComplete) {
      setState(() {
        _animationComplete = false;
      });
      
      // 짧게 확대했다가 축소하는 애니메이션 시퀀스
      Future.delayed(const Duration(milliseconds: 150), () {
        if (mounted) {
          setState(() {
            _animationComplete = true;
          });
        }
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        // 탭 시 애니메이션 시퀀스 실행
        _playTapAnimation();
        widget.onTap();
      },
      onTapDown: (_) => setState(() => _isPressed = true),
      onTapUp: (_) => setState(() => _isPressed = false),
      onTapCancel: () => setState(() => _isPressed = false),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.black,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Expanded(
                child: AnimatedScale(
                  scale: _isPressed || !_animationComplete ? 1.2 : 1.0, // 눌렀을 때 또는 애니메이션 진행 중일 때 1.2배로 확대
                  duration: const Duration(milliseconds: 150), // 애니메이션 지속 시간
                  curve: Curves.easeInOut, // 애니메이션 곡선
                  child: Image.asset(
                    widget.menu.imagePath,
                    fit: BoxFit.contain,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
