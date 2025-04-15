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
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_model.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:dio/dio.dart';
import 'package:open_file/open_file.dart';

class PortfolioEditModal extends ConsumerStatefulWidget {
  final PortfolioModel portfolio;

  const PortfolioEditModal({
    super.key,
    required this.portfolio,
  });

  @override
  ConsumerState<PortfolioEditModal> createState() => _PortfolioEditModalState();
}

class _PortfolioEditModalState extends ConsumerState<PortfolioEditModal> {
  // 상태 변수
  String _category = "";
  String _fileName = ""; // 사용자가 수정 가능한 파일명
  String _originalFileName = ""; // 선택된 파일의 원본 이름
  String _memo = "";
  bool _isSaving = false;
  File? _selectedFile;
  int? _categoryPk;
  bool _isFileUpdated = false;

  @override
  void initState() {
    super.initState();

    // 포트폴리오 데이터로 초기값 설정
    _fileName = widget.portfolio.fileName;
    _originalFileName = widget.portfolio.originFileName;
    _memo = widget.portfolio.portfolioMemo;
    _category = widget.portfolio.portfolioCategory.portfolioCategoryName;
    _categoryPk = widget.portfolio.portfolioCategory.portfolioCategoryPk;

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
        _isFileUpdated = true;
      });

      // 디버그용 로그
      print('Selected file: $filePath');
    }
  }

  // 파일 다운로드
  Future<void> _downloadFile(String fileUrl) async {
    if (fileUrl.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('다운로드할 파일이 없습니다.')),
      );
      return;
    }

    try {
      // Android 13(API 33) 이상에서는 미디어 권한을, 이하 버전에서는 저장소 권한 요청
      bool allPermissionsGranted = false;

      if (await _isAndroid13OrHigher()) {
        // Android 13 이상에서는 photos, videos, audio 권한 요청
        Map<Permission, PermissionStatus> statuses = await [
          Permission.photos,
          Permission.videos,
          Permission.audio,
        ].request();

        allPermissionsGranted =
            statuses.values.every((status) => status.isGranted);
      } else {
        // Android 13 미만에서는 storage 권한 요청
        PermissionStatus status = await Permission.storage.request();
        allPermissionsGranted = status.isGranted;
      }

      if (allPermissionsGranted) {
        setState(() {
          _isSaving = true;
        });

        // 다운로드 디렉토리 가져오기 (Android의 경우 Download 폴더, iOS의 경우 Documents 폴더)
        Directory? directory;
        if (Platform.isAndroid) {
          directory = Directory('/storage/emulated/0/Download');
          // 디렉토리가 없으면 생성
          if (!await directory.exists()) {
            directory = await getExternalStorageDirectory();
          }
        } else if (Platform.isIOS) {
          directory = await getApplicationDocumentsDirectory();
        }

        if (directory == null) {
          throw Exception('저장할 디렉토리를 찾을 수 없습니다.');
        }

        // 파일 이름 준비 (원본 파일명 사용)
        final String fileName = widget.portfolio.originFileName;
        final String filePath = '${directory.path}/$fileName';

        // 파일이 이미 존재하는 경우 삭제
        final file = File(filePath);
        if (await file.exists()) {
          try {
            await file.delete();
            print('기존 파일 삭제 완료: $filePath');
          } catch (e) {
            print('기존 파일 삭제 실패: $e');
            // 파일 삭제 실패 시 임시 파일명으로 저장
            final fileNameWithoutExtension = fileName.split('.').first;
            final fileExtension =
                fileName.contains('.') ? '.${fileName.split('.').last}' : '';
            final newFileName =
                '${fileNameWithoutExtension}_${DateTime.now().millisecondsSinceEpoch}$fileExtension';
            final filePath = '${directory.path}/$newFileName';
          }
        }

        // 로딩 상태 표시
        setState(() {
          _isSaving = true;
        });

        // Dio를 사용하여 파일 다운로드
        final Dio dio = Dio();
        await dio.download(
          fileUrl,
          filePath,
          onReceiveProgress: (received, total) {
            if (total != -1) {
              // 다운로드 진행률 계산 (필요하면 진행률 UI 추가 가능)
              final progress = (received / total * 100).toStringAsFixed(0);
              print('다운로드 진행률: $progress%');
            }
          },
        );

        // 다운로드 완료 메시지
        if (mounted) {
          CompletionMessage.show(context, message: '저장 완료');
          // 바텀 시트 닫기
          Navigator.of(context).pop();
        }
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('파일 다운로드를 위해 저장소 권한이 필요합니다.')),
        );
      }
    } catch (e) {
      print('Download error: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('다운로드 중 오류가 발생했습니다: $e')),
      );
    } finally {
      // 로딩 상태 해제
      if (mounted) {
        setState(() {
          _isSaving = false;
        });
      }
    }
  }

  Future<bool> _isAndroid13OrHigher() async {
    DeviceInfoPlugin deviceInfo = DeviceInfoPlugin();
    AndroidDeviceInfo androidInfo = await deviceInfo.androidInfo;
    return androidInfo.version.sdkInt >= 33; // Android 13은 API 레벨 33
  }

  // 저장 기능
  Future<void> _savePortfolio() async {
    // 필수 필드 검증
    if (_fileName.isEmpty) {
      CompletionMessage.show(context, message: '파일명입력');
      return;
    }

    // 카테고리가 선택되지 않은 경우 -1로 설정 (미분류)
    final categoryPk = _categoryPk ?? -1;

    // 메모가 비어있는 경우 기본값 설정
    final memo = _memo.isEmpty ? " " : _memo;

    // 로딩 상태 설정
    setState(() {
      _isSaving = true;
    });

    try {
      print('Updating portfolio:');
      print('portfolioPk: ${widget.portfolio.portfolioPk}');
      print('file updated: $_isFileUpdated');
      print('portfolioMemo: $memo');
      print('fileName: $_fileName');
      print('portfolioCategoryPk: $categoryPk');

      // 뷰모델을 통해 포트폴리오 수정 요청
      final success =
          await ref.read(portfolioViewModelProvider.notifier).updatePortfolio(
                portfolioPk: widget.portfolio.portfolioPk,
                file: _isFileUpdated
                    ? _selectedFile
                    : null, // 파일이 변경된 경우에만 선택된 파일 전송
                portfolioMemo: memo,
                fileName: _fileName,
                portfolioCategoryPk: categoryPk,
              );

      // 성공적으로 수정된 경우
      if (success) {
        CompletionMessage.show(context, message: '수정 성공');
        // 화면 닫기
        Navigator.of(context).pop(true);
      } else {
        final errorMessage = ref.read(portfolioViewModelProvider).errorMessage;
        print('Update failed with error: $errorMessage');
        CompletionMessage.show(context, message: errorMessage ?? '수정 실패');
      }
    } catch (e, stackTrace) {
      print('Portfolio update error: $e');
      print('Stack trace: $stackTrace');
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

        // 하단 버튼 영역
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: _buildBottomButtons(),
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
              // 원본 파일명 텍스트
              Expanded(
                child: Text(
                  _isFileUpdated
                      ? _originalFileName
                      : widget.portfolio.originFileName,
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
    return Container(
      padding: const EdgeInsets.only(bottom: 100),
      child: Column(
        children: [
          // 카테고리 필드
          PortfolioFields.categoryField(
            context: context,
            ref: ref,
            selectedCategory: _category,
            onCategorySelected: (categoryName, categoryPk) {
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
        ],
      ),
    );
  }

  // 하단 버튼 위젯
  Widget _buildBottomButtons() {
    return Container(
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          // 다운로드 버튼
          Expanded(
            child: Button(
              text: "파일 저장",
              onPressed: _isSaving
                  ? null
                  : () => _downloadFile(widget.portfolio.fileUrl),
              isDisabled: _isSaving,
              textStyle: AppTextStyles.bodyMedium.copyWith(
                color: AppColors.whiteLight,
                fontSize: 14,
              ),
              width: double.infinity,
              height: 50,
            ),
          ),
          const SizedBox(width: 10),
          // 수정하기 버튼
          Expanded(
            child: Button(
              text: "수정하기",
              onPressed: _isSaving ? null : _savePortfolio,
              isDisabled: _isSaving,
              textStyle: AppTextStyles.bodyMedium.copyWith(
                color: AppColors.whiteLight,
                fontSize: 14,
              ),
              width: double.infinity,
              height: 50,
            ),
          ),
        ],
      ),
    );
  }
}
