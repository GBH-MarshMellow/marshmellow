import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/router/routes/cookie_routes.dart';
import 'dart:math';

class LunchCard extends StatefulWidget {
  const LunchCard({super.key});

  @override
  State<LunchCard> createState() => _LunchCardState();
}

class _LunchCardState extends State<LunchCard> {
  // 한식, 중식, 양식 메뉴 리스트
  final List<String> koreanFoods = [
    '김치찌개',
    '청국장',
    '비빔밥',
    '불고기',
    '삼겹살',
    '냉면',
    '덮밥',
    '갈비탕',
    '순두부찌개',
    '제육볶음',
    '국밥',
    '라면'
  ];

  final List<String> chineseFoods = [
    '짜장면',
    '짬뽕',
    '탕수육',
    '마라탕',
    '라멘',
    '탄탄멘',
    '카레',
    '부대찌개',
    '칼국수',
    '스시'
  ];

  final List<String> westernFoods = [
    '파스타',
    '피자',
    '햄버거',
    '서브웨이',
    '샐러드',
    '리조또',
    '샌드위치',
    '떡볶이',
    '치킨',
    '타코'
  ];

  // 모든 메뉴 리스트를 합침
  late List<String> allFoods;

  // 현재 선택된 메뉴
  late String currentFood;
  bool _isLoading = false;
  bool _showMenu = true;

  @override
  void initState() {
    super.initState();
    // 모든 메뉴 리스트 합치기
    allFoods = [...koreanFoods, ...chineseFoods, ...westernFoods];
    // 초기 메뉴 설정
    currentFood = _getRandomFood();
  }

  // 랜덤 메뉴 선택 함수
  String _getRandomFood() {
    final random = Random();
    return allFoods[random.nextInt(allFoods.length)];
  }

  // 메뉴 새로고침 함수
  void _refreshFood() {
    setState(() {
      _showMenu = false;
      _isLoading = true;
    });
    
    // 2초 후에 새 메뉴 표시
    Future.delayed(const Duration(seconds: 2), () {
      setState(() {
        currentFood = _getRandomFood();
        _isLoading = false;
        _showMenu = true;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    final screenHeight = MediaQuery.of(context).size.height;
    final screenWidth = MediaQuery.of(context).size.width;

    return GestureDetector(
      onTap: () => context.push(CookieRoutes.getLunchPath()),
      child: Container(
        height: screenHeight * 0.4,
        decoration: BoxDecoration(
          color: AppColors.yellowPrimary,
          borderRadius: BorderRadius.circular(15),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2), // 그림자 색상 및 투명도
              spreadRadius: 0.5, // 그림자 확산 범위
              blurRadius: 8, // 그림자 흐림 정도
              offset: Offset(0, 4), // 그림자 위치 (x, y)
            ),
          ],
        ),
        child: Stack(
          children: [
            // 제목과 부제목
            Positioned(
              top: 20,
              left: 20,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '점심 메뉴 추천',
                    style: AppTextStyles.mainTitle,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '우리 부서 메뉴 족보로 점심 고민 타파!',
                    style: AppTextStyles.bodyMedium.copyWith(
                      color: AppColors.greyPrimary,
                    ),
                  ),
                ],
              ),
            ),

            // 오른쪽 화살표
            Positioned(
              top: 20,
              right: 20,
              child: SvgPicture.asset(IconPath.caretRight),
            ),

            // 인용부호와 메뉴 텍스트를 담은 컨테이너
            Positioned(
              top: screenHeight * 0.17,
              right: 20, // 오른쪽으로 정렬
              width: screenWidth * 0.6, // 적절한 너비 조정
              child: Row(
                mainAxisAlignment: MainAxisAlignment.end, // 오른쪽 정렬
                children: [
                  Transform.translate(
                    offset: Offset(0, -10), // Y축으로 -10 이동 (위로 올라감)
                    child: SvgPicture.asset(
                      IconPath.quoteLeft,
                      width: 20,
                      height: 20,
                    ),
                  ),
                  // 랜덤 메뉴 텍스트 표시
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 10.0),
                    child: _isLoading
                      ? Transform.scale(
                        scale: 3.0,
                        child: Lottie.asset(
                            'assets/images/loading/loading_lunch.json',
                            width: 80,
                            height: 40,
                            fit: BoxFit.contain,
                          ),
                      )
                      : _showMenu
                        ? Text(
                            currentFood,
                            style: AppTextStyles.modalTitle.copyWith(
                              fontSize: 28,
                              fontWeight: FontWeight.w600,
                            ),
                          )
                        : SizedBox(width: 100, height: 100), // 빈 공간 유지
                  ),
                  Transform.translate(
                    offset: Offset(0, -10), // Y축으로 -10 이동 (위로 올라감)
                    child: SvgPicture.asset(
                      IconPath.quoteRight,
                      width: 20,
                      height: 20,
                    ),
                  ),
                ],
              ),
            ),

            // 캐릭터 이미지
            Positioned(
              bottom: 10,
              left: 20,
              child: Transform(
                alignment: Alignment.center,
                transform: Matrix4.rotationY(3.14159),
                child: Image.asset(
                  'assets/images/characters/char_chair_phone.png',
                  height: screenHeight * 0.175,
                  fit: BoxFit.contain,
                ),
              ),
            ),

            // 새로고침 버튼
            Positioned(
              bottom: 20,
              right: 20,
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.3),
                  shape: BoxShape.circle,
                ),
                child: IconButton(
                  // 버튼 클릭 시 메뉴 새로고침
                  onPressed: _refreshFood,
                  icon: SvgPicture.asset(
                    IconPath.refesh,
                    width: 24,
                    height: 24,
                  ),
                  splashColor: Colors.transparent,
                  highlightColor: Colors.transparent,
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}
