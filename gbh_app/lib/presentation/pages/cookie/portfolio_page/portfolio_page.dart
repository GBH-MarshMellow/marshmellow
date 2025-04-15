import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_category_model.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_model.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_category_item.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_form_modal.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';
import 'package:marshmellow/presentation/viewmodels/portfolio/portfolio_viewmodel.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_item.dart';
import 'package:marshmellow/router/routes/cookie_routes.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/presentation/widgets/completion_message/completion_message.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_delete_confirm.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_edit_modal.dart';
import 'package:marshmellow/presentation/viewmodels/portfolio/portfolio_category_viewmodel.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/category_form_modal.dart';
import 'package:lottie/lottie.dart';

class PortfolioPage extends ConsumerStatefulWidget {
  const PortfolioPage({Key? key}) : super(key: key);

  @override
  ConsumerState<PortfolioPage> createState() => _PortfolioPageState();
}

class _PortfolioPageState extends ConsumerState<PortfolioPage> {
  final TextEditingController _searchController = TextEditingController();

  // 선택 모드 관련 상태 변수
  bool _isSelectionMode = false;
  Set<int> _selectedCategories = {};
  Set<int> _selectedPortfolios = {};

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  void _loadData() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(portfolioViewModelProvider.notifier).loadPortfolios();
      ref.read(portfolioCategoryViewModelProvider.notifier).loadCategories();
    });
  }

  // 포트폴리오 상세 정보 표시
  void _showPortfolioDetail(PortfolioModel portfolio) {
    showModalBottomSheet<bool>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (BuildContext context) {
        return Modal(
          backgroundColor: AppColors.background,
          padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
          child: PortfolioEditModal(
            portfolio: portfolio,
          ),
        );
      },
    ).then((result) {
      // 수정이 완료된 경우 데이터 새로고침
      if (result == true && mounted) {
        _loadData();
      }
    });
  }

  void _showCategoryDetail(PortfolioCategoryModel category) {
    context.push(CookieRoutes.getPortfolioCategoryDetailPath(
        category.portfolioCategoryPk));
  }

  // 카테고리 아이템 선택/해제 토글
  void _toggleCategorySelection(int categoryPk) {
    setState(() {
      if (_selectedCategories.contains(categoryPk)) {
        _selectedCategories.remove(categoryPk);
      } else {
        _selectedCategories.add(categoryPk);
      }
    });
  }

  // 포트폴리오 아이템 선택/해제 토글
  void _togglePortfolioSelection(int portfolioPk) {
    setState(() {
      if (_selectedPortfolios.contains(portfolioPk)) {
        _selectedPortfolios.remove(portfolioPk);
      } else {
        _selectedPortfolios.add(portfolioPk);
      }
    });
  }

  // 선택 모드 시작
  void _startSelectionMode() {
    setState(() {
      _isSelectionMode = true;
    });
  }

  // 선택 모드 종료
  void _endSelectionMode() {
    setState(() {
      _isSelectionMode = false;
      _selectedCategories.clear();
      _selectedPortfolios.clear();
    });
  }

  // 모든 항목 선택
  void _selectAll(List<PortfolioCategoryModel> categories,
      List<PortfolioModel> portfolios) {
    setState(() {
      _selectedCategories = categories
          .where((c) => c.portfolioCategoryName != '미분류')
          .map((c) => c.portfolioCategoryPk)
          .toSet();

      _selectedPortfolios = portfolios
          .where((p) => p.portfolioCategory.portfolioCategoryName == '미분류')
          .map((p) => p.portfolioPk ?? 0)
          .toSet();
    });
  }

  // 모든 항목 선택 해제
  void _deselectAll() {
    setState(() {
      _selectedCategories.clear();
      _selectedPortfolios.clear();
    });
  }

  // 선택한 항목 모두 삭제
  Future<void> _deleteSelected() async {
    if (!mounted) return;

    // 확인 대화상자 표시
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => ConfirmModal(
        title: '알림',
        subtitle: '정말 삭제하시겠습니까?',
        description: '선택한 항목이 영구적으로 삭제되며 \n 복구할 수 없습니다.',
        cancelText: '취소',
        confirmText: '삭제',
      ),
    );

    if (confirm != true || !mounted) return;

    try {
      bool hasErrors = false;

      // 선택된 카테고리 삭제
      if (_selectedCategories.isNotEmpty) {
        final categorySuccess = await ref
            .read(portfolioCategoryViewModelProvider.notifier)
            .deletePortfolioCategories(
                portfolioCategoryPkList: _selectedCategories.toList());

        if (!categorySuccess) hasErrors = true;
      }

      // 선택된 포트폴리오 삭제
      if (_selectedPortfolios.isNotEmpty) {
        final portfolioSuccess = await ref
            .read(portfolioViewModelProvider.notifier)
            .deletePortfolios(portfolioPkList: _selectedPortfolios.toList());

        if (!portfolioSuccess) hasErrors = true;
      }

      if (!mounted) return;

      if (hasErrors) {
        CompletionMessage.show(context, message: '일부 항목 삭제 실패');
      } else {
        CompletionMessage.show(context, message: '삭제 완료');
      }

      // 선택 모드 종료 및 데이터 새로고침
      setState(() {
        _isSelectionMode = false;
        _selectedCategories.clear();
        _selectedPortfolios.clear();
      });

      _loadData();
    } catch (e) {
      if (!mounted) return;
      CompletionMessage.show(context, message: '오류 발생');
    }
  }

  // 개별 포트폴리오 삭제
  Future<void> _deletePortfolio(PortfolioModel portfolio) async {
    if (!mounted) return;

    try {
      final success = await ref
          .read(portfolioViewModelProvider.notifier)
          .deletePortfolios(portfolioPkList: [portfolio.portfolioPk ?? 0]);

      if (!mounted) return;

      if (success) {
        CompletionMessage.show(context, message: '삭제 완료');
      } else {
        CompletionMessage.show(context, message: '삭제 실패');
      }
    } catch (e) {
      if (!mounted) return;
      CompletionMessage.show(context, message: '오류 발생: $e');
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final portfolioState = ref.watch(portfolioViewModelProvider);
    final categoryState = ref.watch(portfolioCategoryViewModelProvider);
    final isLoading = portfolioState.isLoading || categoryState.isLoading;
    final errorMessage =
        portfolioState.errorMessage ?? categoryState.errorMessage;

    // 일반 카테고리와 '미분류' 카테고리 분리
    final regularCategories = categoryState.categories
        .where((category) => category.portfolioCategoryName != '미분류')
        .toList();

    // 미분류 포트폴리오 필터링
    final unclassifiedPortfolios = portfolioState.portfolios
        .where((portfolio) =>
            portfolio.portfolioCategory.portfolioCategoryName == '미분류')
        .toList();

    if (isLoading) {
      return Scaffold(
        body: Center(
          child: Lottie.asset(
            'assets/images/loading/loading_simple.json',
            width: 140,
            height: 140,
            fit: BoxFit.contain,
          ),
        ),
      );
    }

    if (errorMessage != null) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset(
                'assets/images/characters/char_melong.png',
                width: 120,
                height: 120,
              ),
              Text('오류가 발생했습니다.'),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _loadData,
                child: const Text('다시 시도'),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: CustomAppbar(
        title: '포트폴리오',
        actions: [
          if (!_isSelectionMode)
            IconButton(
              onPressed: () {
                // 선택 옵션 보여주기
                showModalBottomSheet<bool>(
                  context: context,
                  backgroundColor: AppColors.background,
                  builder: (context) {
                    return SafeArea(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          ListTile(
                            leading: Icon(Icons.insert_drive_file,
                                color: AppColors.textPrimary),
                            title: Text('포트폴리오 추가',
                                style: AppTextStyles.bodyMedium),
                            onTap: () async {
                              Navigator.pop(context); // 선택 모달 닫기

                              // 포트폴리오 추가 모달 표시
                              final result = await showModalBottomSheet<bool>(
                                context: context,
                                backgroundColor: Colors.transparent,
                                isScrollControlled: true,
                                builder: (BuildContext context) {
                                  return Modal(
                                    backgroundColor: AppColors.background,
                                    padding: const EdgeInsets.symmetric(
                                        vertical: 20, horizontal: 16),
                                    child: const PortfolioForm(),
                                  );
                                },
                              );

                              // 성공적으로 추가된 경우에만 데이터 새로고침
                              if (result == true && mounted) {
                                _loadData();
                              }
                            },
                          ),
                          ListTile(
                            leading: Icon(Icons.folder,
                                color: AppColors.textPrimary),
                            title: Text('카테고리 추가',
                                style: AppTextStyles.bodyMedium),
                            onTap: () async {
                              Navigator.pop(context); // 선택 모달 닫기

                              // 카테고리 추가 모달 표시
                              final result = await showModalBottomSheet<bool>(
                                context: context,
                                backgroundColor: Colors.transparent,
                                isScrollControlled: true,
                                builder: (BuildContext context) {
                                  return Modal(
                                    backgroundColor: AppColors.background,
                                    padding: const EdgeInsets.symmetric(
                                        vertical: 20, horizontal: 16),
                                    child: const CategoryFormModal(),
                                  );
                                },
                              );

                              // 성공적으로 추가된 경우에만 데이터 새로고침
                              if (result == true && mounted) {
                                _loadData();
                              }
                            },
                          ),
                        ],
                      ),
                    );
                  },
                );
              },
              icon: SvgPicture.asset(IconPath.add),
            ),
        ],
      ),
      body: Column(
        children: [
          // 선택 모드 컨트롤바
          if (_isSelectionMode)
            Container(
              height: 60,
              color: AppColors.background,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  // 닫기 버튼
                  TextButton(
                    onPressed: _endSelectionMode,
                    child: Text(
                      '닫기',
                      style: AppTextStyles.bodySmall.copyWith(
                        color: AppColors.textPrimary,
                      ),
                    ),
                  ),

                  // 전체 선택 버튼
                  TextButton(
                    onPressed: () =>
                        _selectAll(regularCategories, unclassifiedPortfolios),
                    child: Text(
                      '전체 선택',
                      style: AppTextStyles.bodySmall.copyWith(
                        color: AppColors.textPrimary,
                      ),
                    ),
                  ),

                  // 전체 해제 버튼
                  TextButton(
                    onPressed: _deselectAll,
                    child: Text(
                      '전체 해제',
                      style: AppTextStyles.bodySmall.copyWith(
                        color: AppColors.textPrimary,
                      ),
                    ),
                  ),

                  // 삭제 버튼
                  IconButton(
                    onPressed: _selectedCategories.isNotEmpty ||
                            _selectedPortfolios.isNotEmpty
                        ? _deleteSelected
                        : null,
                    icon: SvgPicture.asset(
                      IconPath.trash,
                      width: 20,
                      height: 20,
                    ),
                  ),
                ],
              ),
            ),

          // 스크롤 가능한 콘텐츠 영역
          Expanded(
            child: SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 카테고리 그리드
                    if (regularCategories.isNotEmpty) ...[
                      GridView.builder(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        gridDelegate:
                            const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 2,
                          crossAxisSpacing: 8,
                          mainAxisSpacing: 8,
                          childAspectRatio: 300 / 260,
                        ),
                        itemCount: regularCategories.length,
                        itemBuilder: (context, index) {
                          final category = regularCategories[index];
                          final isSelected = _selectedCategories
                              .contains(category.portfolioCategoryPk);

                          return CategoryItem(
                            category: category,
                            isSelected: _selectedCategories
                                .contains(category.portfolioCategoryPk),
                            isSelectionMode: _isSelectionMode,
                            onSelectionToggle: () {
                              _toggleCategorySelection(
                                  category.portfolioCategoryPk);
                            },
                            onTap: () {
                              context
                                  .push(CookieRoutes
                                      .getPortfolioCategoryDetailPath(
                                          category.portfolioCategoryPk))
                                  .then((_) {
                                ref
                                    .read(portfolioCategoryViewModelProvider
                                        .notifier)
                                    .loadCategories();
                              });
                            },
                            onLongPress: () {
                              if (!_isSelectionMode) {
                                _startSelectionMode();
                                _toggleCategorySelection(
                                    category.portfolioCategoryPk);
                              }
                            },
                          );
                        },
                      ),
                      const SizedBox(height: 24),
                    ],

                    // 전체가 비어있는 경우 (카테고리도 없고 미분류 포트폴리오도 없음)
                    if (regularCategories.isEmpty &&
                        unclassifiedPortfolios.isEmpty &&
                        !_isSelectionMode)
                      Center(
                        child: Column(
                          children: [
                            Image.asset(
                              'assets/images/characters/char_melong.png',
                              width: 120,
                              height: 120,
                            ),
                            const SizedBox(height: 16),
                            Text(
                              '아직 등록된 포트폴리오가 없습니다.\n우측 상단의 + 버튼을 눌러\n포트폴리오나 카테고리를 추가해보세요.',
                              style: AppTextStyles.bodyMedium,
                              textAlign: TextAlign.center,
                            ),
                          ],
                        ),
                      )
                    // 카테고리는 있지만 미분류 포트폴리오가 없는 경우
                    else if (unclassifiedPortfolios.isEmpty &&
                        !_isSelectionMode &&
                        regularCategories.isNotEmpty)
                      // 아무 내용도 표시하지 않음 (카테고리 그리드만 표시)
                      const SizedBox()
                    else if (unclassifiedPortfolios.isNotEmpty)
                      ListView.builder(
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        itemCount: unclassifiedPortfolios.length,
                        itemBuilder: (context, index) {
                          final portfolio = unclassifiedPortfolios[index];
                          final isSelected = _selectedPortfolios
                              .contains(portfolio.portfolioPk);

                          return PortfolioItem(
                            portfolio: portfolio,
                            isSelected: isSelected,
                            isSelectionMode: _isSelectionMode,
                            onSelectionToggle: () => _togglePortfolioSelection(
                                portfolio.portfolioPk ?? 0),
                            onDelete: (portfolio) =>
                                _deletePortfolio(portfolio),
                            onTap: () {
                              if (_isSelectionMode) {
                                _togglePortfolioSelection(
                                    portfolio.portfolioPk ?? 0);
                              } else {
                                _showPortfolioDetail(portfolio);
                              }
                            },
                            onLongPress: () {
                              if (!_isSelectionMode) {
                                _startSelectionMode();
                                _togglePortfolioSelection(
                                    portfolio.portfolioPk ?? 0);
                              }
                            },
                          );
                        },
                      ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
