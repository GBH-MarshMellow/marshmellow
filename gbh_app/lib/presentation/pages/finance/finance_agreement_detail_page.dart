// finance_agreement_detail_page.dart
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';

class FinanceAgreementDetailPage extends StatefulWidget {
  final String agreementNo;

  const FinanceAgreementDetailPage({
    Key? key,
    required this.agreementNo,
  }) : super(key: key);

  @override
  State<FinanceAgreementDetailPage> createState() => _FinanceAgreementDetailPageState();
}

class _FinanceAgreementDetailPageState extends State<FinanceAgreementDetailPage> {
  String _agreementContent = "약관 내용을 불러오는 중...";
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadAgreementContent();
  }

  Future<void> _loadAgreementContent() async {
    try {
      // assets 폴더에서 마크다운 파일 로드
      final String content = await rootBundle.loadString(
        'assets/agreements/${widget.agreementNo}.md',
      );
      
      setState(() {
        _agreementContent = content;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _agreementContent = "약관 내용을 불러올 수 없습니다.";
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppbar(
        title: getAgreementTitle(widget.agreementNo),
        backgroundColor: AppColors.divider,
      ),
      body: Column(
        children: [
          // 약관 내용 (스크롤 가능)
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : SingleChildScrollView(
                    padding: const EdgeInsets.all(16.0),
                    child: MarkdownBody(
                      data: _agreementContent,
                      styleSheet: MarkdownStyleSheet(
                        p: const TextStyle(fontSize: 14, height: 1.5),
                        h1: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                        h2: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
          ),
          
          // 확인 버튼
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Button(
              text: '확인',
              onPressed: () {
                Navigator.pop(context);
              },
            ),
          ),
        ],
      ),
    );
  }
  
  // agreementNo에 따라 약관 제목 반환
  String getAgreementTitle(String agreementNo) {
    switch (agreementNo) {
      case 'A001':
        return '오픈뱅킹 서비스 이용약관';
      case 'A002':
        return '고객본인확인';
      // 다른 약관 케이스...
      default:
        return '약관 상세';
    }
  }
}