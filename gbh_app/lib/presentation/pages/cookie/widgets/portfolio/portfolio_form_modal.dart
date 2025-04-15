import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_fields.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io';
import 'package:marshmellow/presentation/viewmodels/portfolio/portfolio_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/completion_message/completion_message.dart';
import 'package:marshmellow/presentation/viewmodels/portfolio/portfolio_category_viewmodel.dart';

class PortfolioForm extends ConsumerStatefulWidget {
  final String fileName;
  final int? initialCategoryPk;
  final String? initialCategoryName;

  const PortfolioForm({
    super.key,
    this.fileName = "", // 기본 파일명
    this.initialCategoryPk, // 초기 카테고리 PK
    this.initialCategoryName, // 초기 카테고리 이름
  });

  @override
  ConsumerState<PortfolioForm> createState() => _PortfolioFormState();
}

class _PortfolioFormState extends ConsumerState<PortfolioForm> {
  // 상태 변수
  String _category = "";
  String _fileName = ""; // 사용자가 수정 가능한 파일명
  String _originalFileName = ""; // 선택된 파일의 원본 이름
  String _memo = "";
  DateTime _selectedDate = DateTime.now();
  bool _isSaving = false;
  File? _selectedFile;
  int? _categoryPk;

  @override
  void initState() {
    super.initState();
    // 파일명 초기화
    _fileName = widget.fileName;
    _originalFileName = widget.fileName;

    // 카테고리 초기화 (전달된 경우)
    if (widget.initialCategoryName != null) {
      _category = widget.initialCategoryName!;
    }
    if (widget.initialCategoryPk != null) {
      _categoryPk = widget.initialCategoryPk;
    }

    // 포트폴리오 카테고리 목록 불러오기
    _loadCategories();
  }

  // 카테고리 목록 불러오기
  Future<void> _loadCategories() async {
    // 중복 로드 방지를 위한 조건 추가
    if (!ref.read(portfolioCategoryViewModelProvider).isLoading &&
        ref.read(portfolioCategoryViewModelProvider).categories.isEmpty) {
      await ref
          .read(portfolioCategoryViewModelProvider.notifier)
          .loadCategories();
    }
  }

  // 파일명 업데이트
  void _updateFileName(String fileName) {
    setState(() {
      _fileName = fileName;
    });
  }

  // 메모 업데이트
  void _updateMemo(String memo) {
    setState(() {
      _memo = memo;
    });
  }

  // 날짜 업데이트
  void _updateDate(DateTime date) {
    setState(() {
      _selectedDate = date;
    });
  }

  // 파일 선택
  Future<void> _selectFile() async {
    // 파일 선택기 열기
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      allowMultiple: false,
    );

    if (result != null && result.files.single.path != null) {
      // 선택된 파일 경로
      String filePath = result.files.single.path!;

      // 파일 객체 생성 및 파일명 업데이트
      setState(() {
        _selectedFile = File(filePath);
        _originalFileName = result.files.single.name; // 헤더에 표시할 원본 파일명
        _fileName = result.files.single.name; // 사용자가 수정할 수 있는 필드에도 초기값 설정
      });

      // 디버그용 로그
      print('Selected file: $filePath');
    }
  }

  // 저장 기능
  Future<void> _savePortfolio() async {
    // 필수 필드 검증
    if (_selectedFile == null) {
      CompletionMessage.show(context, message: '파일 선택');
      return;
    }

    if (_fileName.isEmpty) {
      CompletionMessage.show(context, message: '파일명');
      return;
    }

    // 카테고리가 선택되지 않은 경우 -1로 설정 (미분류)
    final categoryPk = _categoryPk ?? -1;

    // 로딩 상태 설정
    setState(() {
      _isSaving = true;
    });

    try {
      // 뷰모델을 통해 포트폴리오 등록 요청
      final portfolio =
          await ref.read(portfolioViewModelProvider.notifier).createPortfolio(
                file: _selectedFile!,
                portfolioMemo: _memo,
                fileName: _fileName,
                portfolioCategoryPk: categoryPk,
              );

      // 성공적으로 등록된 경우
      if (portfolio != null) {
        CompletionMessage.show(context, message: '등록 성공');
        // 화면 닫기
        Navigator.of(context).pop(true);
      } else {
        final errorMessage = ref.read(portfolioViewModelProvider).errorMessage;
        CompletionMessage.show(context, message: errorMessage ?? '등록 실패');
      }
    } catch (e) {
      CompletionMessage.show(context, message: '오류 발생: $e');
    } finally {
      // 로딩 상태 해제
      if (mounted) {
        setState(() {
          _isSaving = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // 메인 콘텐츠
        SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // 파일 정보 헤더
              _buildFileHeader(),

              const Divider(
                  height: 1, thickness: 1, color: AppColors.greyLight),

              // 폼 필드들
              _buildFormFields(context),
            ],
          ),
        ),

        // 하단 저장 버튼
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: _buildSaveButton(),
        ),
      ],
    );
  }

  // 파일 정보 헤더 위젯
  Widget _buildFileHeader() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // 원본 파일명 텍스트 (헤더에 표시)
              Expanded(
                child: Text(
                  _originalFileName.isEmpty ? "파일 선택" : _originalFileName,
                  style: AppTextStyles.modalTitle.copyWith(
                    fontWeight: FontWeight.bold,
                    fontSize: 22,
                  ),
                  overflow: TextOverflow.ellipsis,
                  maxLines: 1,
                ),
              ),
              // 파일 선택 아이콘
              GestureDetector(
                onTap: _selectFile,
                child: SvgPicture.asset(
                  IconPath.paperclip,
                  width: 20,
                  height: 20,
                  colorFilter: ColorFilter.mode(
                    AppColors.textPrimary,
                    BlendMode.srcIn,
                  ),
                ),
              ),
              const SizedBox(width: 10),
            ],
          ),
        ],
      ),
    );
  }

  // 폼 필드들 위젯
  Widget _buildFormFields(BuildContext context) {
    // 포트폴리오 상태 가져오기
    final portfolioState = ref.watch(portfolioViewModelProvider);

    return Container(
      padding: const EdgeInsets.only(bottom: 100),
      child: Column(
        children: [
          // 카테고리 필드
          PortfolioFields.categoryField(
            context: context,
            ref: ref,
            selectedCategory: _category, // 이미 선택된 카테고리
            onCategorySelected: (categoryName, categoryPk) {
              // setState를 통해 상태만 업데이트하고, 다른 메서드 호출은 피합니다
              setState(() {
                _category = categoryName;
                _categoryPk = categoryPk;
              });
            },
          ),
          // 파일명 필드
          PortfolioFields.editableFileNameField(
            fileName: _fileName,
            onFileNameChanged: _updateFileName,
          ),

          // 메모/키워드 필드
          PortfolioFields.editableMemoField(
            memo: _memo,
            onMemoChanged: _updateMemo,
          ),

          // 날짜 필드
          PortfolioFields.dateField(
            context: context,
            ref: ref,
            selectedDate: _selectedDate,
            onDateChanged: _updateDate,
          ),
        ],
      ),
    );
  }

  // 저장 버튼 위젯
  Widget _buildSaveButton() {
    return Container(
      padding: const EdgeInsets.all(16),
      child: Button(
        text: "저장하기",
        onPressed: _isSaving ? null : _savePortfolio,
        width: double.infinity,
        height: 50,
      ),
    );
  }
}
