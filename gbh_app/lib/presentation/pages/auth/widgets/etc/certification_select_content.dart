import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/di/providers/modal_provider.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/certification_card.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';

/*
  mm인증서 선택 모달
*/
// 인증서 모달 표시 함수
void showCertificateModal({
  required BuildContext context,
  required WidgetRef ref,
  required String userName,
  required String expiryDate,
  required VoidCallback onConfirm,
  String title = '인증서 로그인'
}) {
  // 모달 상태 변경
  ref.read(modalProvider.notifier).showModal();

  showModalBottomSheet(
    context: context, 
    backgroundColor: Colors.transparent,
    isScrollControlled: true,
    builder: (BuildContext context) {
      return Modal(
        backgroundColor: AppColors.whiteLight, 
        padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
        title: title,
        titleStyle: AppTextStyles.bodyMediumLight,
        showDivider: false,
        child: CertificateSelectContent(
          userName: userName,
          expiryDate: expiryDate,
          onConfirm: () {
            Navigator.pop(context);
            onConfirm();
          }
        ),
      );
    }
  ).then((_) {
    // 모달 닫힐 때 상태 업데이트
    ref.read(modalProvider.notifier).hideModal();
  });
}

class CertificateSelectContent extends StatelessWidget {
  final String userName;
  final String expiryDate;
  final VoidCallback onConfirm;

  const CertificateSelectContent({
    Key? key,
    required this.userName,
    required this.expiryDate,
    required this.onConfirm,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final screenWidth = MediaQuery.of(context).size.width;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 16),
            child: SmallCertificateCard(
              userName: userName, 
              expiryDate: expiryDate,
              onTap: onConfirm,
            ),
          )
        ),

        const SizedBox(height: 24),

        // 다음 버튼
        Button(
          text: '다음',
          width: screenWidth * 0.9,
          height: 60,
          onPressed: onConfirm,
        )
      ],
    );
  }
}