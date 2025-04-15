// 점심 메뉴 데이터 모델 정의
class LunchMenu {
  final String id;       // 메뉴 고유 ID
  final String name;     // 메뉴 이름
  final String imagePath; // 메뉴 이미지 경로

  const LunchMenu({
    required this.id,
    required this.name,
    required this.imagePath,
  });
}

// 모든 점심 메뉴 목록
final List<LunchMenu> allLunchMenus = [
  const LunchMenu(
    id: 'chinese',
    name: '짜장면',
    imagePath: 'assets/images/food/chinese.png',
  ),
  const LunchMenu(
    id: 'japanese',
    name: '초밥',
    imagePath: 'assets/images/food/japanese.png',
  ),
  const LunchMenu(
    id: 'western',
    name: '타코',
    imagePath: 'assets/images/food/taco.png',
  ),
  const LunchMenu(
    id: 'snack',
    name: '떡볶이',
    imagePath: 'assets/images/food/snack.png',
  ),
  const LunchMenu(
    id: 'fastfood',
    name: '햄버거',
    imagePath: 'assets/images/food/fastfood.png',
  ),
  const LunchMenu(
    id: 'salad',
    name: '샐러드',
    imagePath: 'assets/images/food/salad.png',
  ),
  const LunchMenu(
    id: 'dosirak',
    name: '도시락',
    imagePath: 'assets/images/food/dosirak.png',
  ),
  const LunchMenu(
    id: 'gukbap',
    name: '국밥',
    imagePath: 'assets/images/food/gukbap.png',
  ),
  const LunchMenu(
    id: 'stew',
    name: '돼지찌개',
    imagePath: 'assets/images/food/stew.png',
  ),
  const LunchMenu(
    id: 'baekban',
    name: '집밥',
    imagePath: 'assets/images/food/baekban.png',
  ),
  const LunchMenu(
    id: 'deopbap',
    name: '차슈덮밥',
    imagePath: 'assets/images/food/deopbap.png',
  ),
  const LunchMenu(
    id: 'samgyeopsal',
    name: '삼겹살',
    imagePath: 'assets/images/food/samgyeopsal.png',
  ),
  const LunchMenu(
    id: 'jeyukbokkeum',
    name: '제육볶음',
    imagePath: 'assets/images/food/jeyukbokkeum.png',
  ),
  const LunchMenu(
    id: 'pho',
    name: '쌀국수',
    imagePath: 'assets/images/food/pho.png',
  ),
  const LunchMenu(
    id: 'maratang',
    name: '마라탕',
    imagePath: 'assets/images/food/maratang.png',
  ),
  const LunchMenu(
    id: 'cafeteria',
    name: '구내식당',
    imagePath: 'assets/images/food/cafeteria.png',
  ),
];