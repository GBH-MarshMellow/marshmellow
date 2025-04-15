import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';

import 'package:marshmellow/core/constants/icon_path.dart';

/*
  상단 커스텀 앱바 UI
*/
class CustomAppbar extends StatelessWidget implements PreferredSizeWidget {
  final String title;
  final List<Widget> actions;
  final VoidCallback? onNofiticationTap;
  final Color backgroundColor;
  final bool hideNotificaiotnIcon;
  final bool automaticallyImplyLeading; // 알림페이지에서는 알림 아이콘을 표시하지 않을겁니다

  const CustomAppbar({
    Key? key,
    required this.title,
    this.actions = const [],
    this.onNofiticationTap,
    this.backgroundColor = AppColors.background,
    this.hideNotificaiotnIcon = false,
    this.automaticallyImplyLeading = true,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return AppBar(
      backgroundColor: backgroundColor,
      title: Text(title, style: AppTextStyles.appBar),
      scrolledUnderElevation: 0, // 스크롤 시 그림자/색상 변화 비활성화
      elevation: 0, // 그림자 효과 제거
      automaticallyImplyLeading: automaticallyImplyLeading, // 자동 뒤로가기 버튼 비활성화
      actions: [
        // 아이콘을 추가할 거라면 여기에 넣으세요
        ...actions,
        // 알림 아이콘 우측 고정
        if (!hideNotificaiotnIcon)
          IconButton(
            icon: SvgPicture.asset(IconPath.bell),
            constraints: BoxConstraints(),
            padding: const EdgeInsets.only(left: 0, right: 12),
            // 누르면 알림 페이지로 이동
            onPressed: onNofiticationTap ??
                () {
                  GoRouter.of(context).push('/notification');
                },
          )
      ],
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}
