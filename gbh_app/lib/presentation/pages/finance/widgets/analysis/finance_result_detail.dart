import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/finance/finance_type_model.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:screenshot/screenshot.dart';
import 'package:share_plus/share_plus.dart';
import 'package:path_provider/path_provider.dart';

// 유형 결과 상세 정보 위젯
class FinanceResultDetail extends StatefulWidget {
  final FinanceTypeModel financeType;

  const FinanceResultDetail({
    Key? key,
    required this.financeType,
  }) : super(key: key);

  @override
  State<FinanceResultDetail> createState() => _FinanceResultDetailState();
}

class _FinanceResultDetailState extends State<FinanceResultDetail> {
  // 스크린샷 컨트롤러를 state 변수로 선언
  final ScreenshotController screenshotController = ScreenshotController();

  @override
  Widget build(BuildContext context) {
    // 스크린샷 컨트롤러 생성
    // final screenshotController = ScreenshotController();
    
    return Column(
      children: [
        // 스크린샷으로 캡처할 영역
        Screenshot(
          controller: screenshotController,
          child: Container(
            padding: const EdgeInsets.all(16),
            color: AppColors.background,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // 타이틀 헤더 추가
                // Center(
                //   child: Text(
                //     'Marshmellow 자산유형 분석',
                //     style: const TextStyle(
                //       fontSize: 12,
                //       fontWeight: FontWeight.w400,
                //       color: AppColors.disabled,
                //     ),
                //   ),
                // ),
                // const SizedBox(height: 32),
                
                // 유형 및 닉네임
                Text(
                  '당신은',
                  style: const TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: AppColors.blackPrimary
                  ),
                  textAlign: TextAlign.center,
                ),
                Text(
                  widget.financeType.nickname,
                  style: const TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: AppColors.blackPrimary
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                
                // 유형 이미지
                Center(
                  child: Image.asset(
                    widget.financeType.imagePath,
                    height: 320,
                    width: double.infinity,
                    fit: BoxFit.contain,
                  ),
                ),
                
                const SizedBox(height: 8),
                
                Center(
                  child: Text(
                    widget.financeType.shortDescription,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: AppColors.blackPrimary
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
                
                const SizedBox(height: 24),
                
                Text(
                  widget.financeType.longDescription,
                  style: AppTextStyles.bodyMediumLight.copyWith(
                    letterSpacing: 0.5,
                    height: 1.6,
                  ),
                  softWrap: true,
                  textAlign: TextAlign.left,
                ),
              ],
            ),
          ),
        ),
        
        // 공유하기 버튼 - 하단에 배치
        const SizedBox(height: 20),
        InkWell(
          onTap: () => _shareScreenshot(context),
          child: Button(
            text: '공유하기',
            borderRadius: 10,
            width: MediaQuery.of(context).size.width * 0.25,
            height: MediaQuery.of(context).size.width * 0.12,
          ),
        ),
      ],
    );
  }

  // 스크린샷 캡처 및 공유 메서드 - 매개변수 수정
  Future<void> _shareScreenshot(BuildContext context) async {
    try {
      // 로딩 표시
      final scaffold = ScaffoldMessenger.of(context);
      scaffold.showSnackBar(
        const SnackBar(content: Text('이미지 생성 중...'), duration: Duration(seconds: 1)),
      );

      // 스크린샷 캡처
      final Uint8List? imageBytes = await screenshotController.capture();
      if (imageBytes == null) {
        throw Exception('이미지 캡처에 실패했습니다');
      }

      // 임시 파일 생성
      final directory = await getTemporaryDirectory();
      final imagePath = '${directory.path}/finance_type_${widget.financeType.id}.png';  // widget. 추가
      final imageFile = File(imagePath);
      await imageFile.writeAsBytes(imageBytes);

      // 공유하기 실행
      await Share.shareXFiles(
        [XFile(imagePath)],
        text: '마시멜로 자산유형 분석 결과: ${widget.financeType.nickname}',  // widget. 추가
        subject: '나의 자산유형 분석 결과',
      );
    } catch (e) {
      // 오류 처리
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('공유하기 실패: ${e.toString()}')),
      );
    }
  }
}