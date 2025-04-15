import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/presentation/widgets/round_input/round_input.dart';

class CustomSearchBar extends StatelessWidget {
  final TextEditingController controller;
  final ValueChanged<String>? onChanged;
  final VoidCallback? onSearchPressed;
  final ValueChanged<String>? onSubmitted; // 타입 변경
  final String? hintText;
  final double? height;
  final FocusNode? focusNode;

  const CustomSearchBar(
      {Key? key,
      required this.controller,
      this.onChanged,
      this.onSearchPressed,
      this.hintText,
      this.height,
      this.onSubmitted, // 타입 변경
      this.focusNode})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    final containerWidth = MediaQuery.of(context).size.width * 0.9;

    return SizedBox(
      height: 50, // 높이 설정
      width: containerWidth,
      child: Stack(
        children: [
          // RoundInput을 기본 배경으로 사용
          Positioned(
            left: 0,
            top: 0,
            child: RoundInput(
              width: containerWidth * 0.85,
              height: height ?? 40, // 높이 설정
              controller: controller,
              focusNode: focusNode,
              onChanged: onChanged,
              onSubmitted: onSubmitted,
              hintText: hintText,
              showDropdown: false,
            ),
          ),

          // 검색 아이콘
          Positioned(
            right: 0,
            top: 0,
            child: GestureDetector(
              onTap: onSearchPressed,
              child: SvgPicture.asset(
                'assets/icons/search_bar/search_button.svg',
                width: 40,
                height: 40,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
