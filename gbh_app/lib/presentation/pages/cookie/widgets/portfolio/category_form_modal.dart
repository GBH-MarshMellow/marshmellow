import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/category_fields.dart';
import 'package:marshmellow/presentation/viewmodels/portfolio/portfolio_category_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/completion_message/completion_message.dart';

class CategoryFormModal extends ConsumerStatefulWidget {
  const CategoryFormModal({Key? key}) : super(key: key);

  @override
  ConsumerState<CategoryFormModal> createState() => _CategoryFormModalState();
}

class _CategoryFormModalState extends ConsumerState<CategoryFormModal> {
  // 상태 변수
  String _categoryName = "";
  String _memo = "";
  bool _isSaving = false;

  // 카테고리명 업데이트
  void _updateCategoryName(String name) {
    setState(() {
      _categoryName = name;
    });
  }

  // 메모 업데이트
  void _updateMemo(String memo) {
    setState(() {
      _memo = memo;
    });
  }

  // 카테고리 저장
  Future<void> _saveCategory() async {
    // 필드에서 포커스 해제하여 현재 편집 중인 텍스트 값을 반영
    FocusScope.of(context).unfocus();

    // 약간의 지연을 주어 onFocusChange 콜백이 실행될 시간을 확보
    await Future.delayed(const Duration(milliseconds: 100));

    // 필수 필드 검증
    if (_categoryName.isEmpty) {
      CompletionMessage.show(context, message: '카테고리명을 입력해주세요');
      return;
    }

    // 로깅 추가
    print('저장 시도: 카테고리명=$_categoryName, 메모=$_memo');

    // 로딩 상태 설정
    setState(() {
      _isSaving = true;
    });

    try {
      // 카테고리 생성 요청
      final success = await ref
          .read(portfolioCategoryViewModelProvider.notifier)
          .createPortfolioCategory(
            categoryName: _categoryName,
            categoryMemo: _memo,
          );

      print('카테고리 저장 결과: $success');

      if (mounted) {
        if (success) {
          CompletionMessage.show(context, message: '등록 완료');
          Navigator.of(context).pop(true); // 성공 결과 반환하며 닫기
        } else {
          final errorMessage =
              ref.read(portfolioCategoryViewModelProvider).errorMessage;
          CompletionMessage.show(context, message: errorMessage ?? '등록 실패');
        }
      }
    } catch (e) {
      print('카테고리 저장 오류: $e');
      if (mounted) {
        CompletionMessage.show(context, message: '오류 발생: $e');
      }
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
              // 제목
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 20, 16, 10),
                child: Text(
                  '카테고리 등록',
                  style: AppTextStyles.modalTitle.copyWith(
                    fontWeight: FontWeight.bold,
                    fontSize: 22,
                  ),
                ),
              ),

              const Divider(
                height: 1,
                thickness: 1,
                color: AppColors.greyLight,
              ),

              // 폼 필드들
              Container(
                padding: const EdgeInsets.only(bottom: 100),
                child: Column(
                  children: [
                    // 카테고리명 필드
                    CategoryFields.editableCategoryNameField(
                      categoryName: _categoryName,
                      onCategoryNameChanged: _updateCategoryName,
                    ),

                    // 메모 필드
                    CategoryFields.editableMemoField(
                      memo: _memo,
                      onMemoChanged: _updateMemo,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),

        // 하단 저장 버튼
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: Container(
            padding: const EdgeInsets.all(16),
            child: Button(
              text: "저장하기",
              onPressed: _isSaving ? null : _saveCategory,
              width: double.infinity,
              height: 50,
            ),
          ),
        ),
      ],
    );
  }
}
