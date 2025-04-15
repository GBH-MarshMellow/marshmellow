import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/card/card.dart';
import 'package:marshmellow/presentation/widgets/round_input/round_input.dart';
import 'package:marshmellow/presentation/widgets/select_input/select_input.dart';
import 'package:marshmellow/presentation/widgets/text_input/text_input.dart';
import 'package:marshmellow/presentation/widgets/custom_search_bar/custom_search_bar.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/custom_search_bar/custom_search_bar.dart';

class InputPage extends StatefulWidget {
  const InputPage({super.key});

  @override
  State<InputPage> createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _countryController = TextEditingController();
  final TextEditingController _roundController = TextEditingController();
  final TextEditingController _categoryController = TextEditingController();
  final TextEditingController _searchController = TextEditingController();

  String? _emailError;

  @override
  void initState() {
    super.initState();
    _emailController.addListener(_validateEmail);
  }

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _countryController.dispose();
    _roundController.dispose();
    _categoryController.dispose();
    _searchController.dispose();
    super.dispose();
  }

  void _validateEmail() {
    final email = _emailController.text;

    if (email.isEmpty) {
      setState(() {
        _emailError = null;
      });
      return;
    }

    // 이메일 유효성 검증 정규식
    final emailRegex =
        RegExp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$');

    if (!emailRegex.hasMatch(email)) {
      setState(() {
        _emailError = '유효한 이메일 주소를 입력해주세요';
      });
    } else {
      setState(() {
        _emailError = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('입력 페이지'),
      ),
      body: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CustomSearchBar(
              controller: _searchController,
              onSearchPressed: () {
                print('검색 버튼 클릭됨');
              },
            ),
            const SizedBox(height: 20),
            CustomCard(
              width: MediaQuery.of(context).size.width * 0.9,
              height: MediaQuery.of(context).size.height * 0.25,
              backgroundColor: AppColors.bluePrimary,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '제목',
                    style: AppTextStyles.bodySmall,
                  ),
                  SizedBox(height: 8),
                  Text(
                    '여기에 카드 내용을 적습니다. 여러 줄의 텍스트와 다양한 위젯을 포함할 수 있습니다.',
                    style: AppTextStyles.bodyMedium
                        .copyWith(fontWeight: FontWeight.w300),
                  ),
                ],
              ),
              onTap: () {
                print('카드가 탭됨');
              },
            ),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                CustomCard(
                  width: MediaQuery.of(context).size.width * 0.45,
                  height: MediaQuery.of(context).size.height * 0.13,
                  backgroundColor: AppColors.pinkPrimary,
                  onTap: () {
                    print('카드가 탭됨');
                  },
                  child: const Text('카드 내용'),
                ),
                SizedBox(width: MediaQuery.of(context).size.width * 0.01),
                CustomCard(
                  width: MediaQuery.of(context).size.width * 0.45,
                  height: MediaQuery.of(context).size.height * 0.13,
                  backgroundColor: AppColors.bluePrimary,
                  onTap: () {
                    print('카드가 탭됨');
                  },
                  child: const Text('카드 내용'),
                ),
              ],
            ),
            const SizedBox(height: 20),
            SelectInput<String>(
              label: "국가 선택",
              modalTitle: "국가 선택",
              controller: _countryController,
              items: ["미국", "캐나다", "영국", "호주", "일본", "한국"],
              itemBuilder: (context, item, isSelected) =>
                  ListTile(title: Text(item)),
              onItemSelected: (value) {
                print("선택된 국가: $value");
              },
            ),
            const SizedBox(height: 20),
            TextInput(
              label: '이메일',
              controller: _emailController,
              errorText: _emailError,
            ),
            const SizedBox(height: 30),
            RoundInput(
              height: 50,
              onChanged: (value) {
                print('이름: $value');
              },
            ),
            const SizedBox(height: 10),
            RoundInput(
              hintText: '입금자명 4글자',
              height: 50,
              onChanged: (value) {
                print('이름: $value');
              },
            ),
            const SizedBox(height: 10),
            RoundInput(
              label: '입금자명',
              hintText: '입금자명 4글자',
              controller: _roundController,
              height: 50,
              onChanged: (value) {
                print('이름: $value');
              },
            ),
            const SizedBox(height: 10),

            RoundInput<String>(
              label: "카테고리",
              hintText: "카테고리를 선택하세요",
              controller: _categoryController,
              showDropdown: true,
              modalTitle: "카테고리 선택",
              items: [
                '식사',
                '카페/디저트',
                '쇼핑',
                '문화/여가',
                '교통',
              ],
              onItemSelected: (value) {
                print("선택된 국가: $value");
              },
            ),
            const SizedBox(height: 10),
            Button(
              onPressed: () {
                _validateEmail(); // 버튼 클릭 시 한번 더 검증
                if (_emailError == null && _emailController.text.isNotEmpty) {
                  print(
                      '폼 제출: 이름=${_nameController.text}, 이메일=${_emailController.text}, 국가=${_countryController.text}');
                }
              },
            ),
            const SizedBox(height: 10),
            Button(
              text: '확인',
              borderRadius: 30,
              onPressed: () {
                print('확인');
              },
            ),
            const SizedBox(height: 10),
            // 네/아니오 버튼 가로로 나란히 배치
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Button(
                  text: '네',
                  width: MediaQuery.of(context).size.width * 0.43, // 화면 너비의 43%
                  borderRadius: 30,
                  onPressed: () {
                    print('네 선택됨');
                  },
                ),
                const SizedBox(width: 10), // 버튼 사이 간격
                Button(
                  text: '아니오',
                  width: MediaQuery.of(context).size.width * 0.43, // 화면 너비의 43%
                  borderRadius: 30,
                  onPressed: () {
                    print('아니오 선택됨');
                  },
                ),
              ],
            ),
            const SizedBox(height: 10),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Button(
                  text: '확인',
                  width: MediaQuery.of(context).size.width * 0.43, // 화면 너비의 43%
                  onPressed: () {
                    print('네 선택됨');
                  },
                ),
                const SizedBox(width: 10), // 버튼 사이 간격
                Button(
                  text: '취소',
                  width: MediaQuery.of(context).size.width * 0.43, // 화면 너비의 43%
                  onPressed: () {
                    print('아니오 선택됨');
                  },
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
