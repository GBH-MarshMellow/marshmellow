import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_slidable/flutter_slidable.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_item.dart';
import 'package:marshmellow/presentation/viewmodels/portfolio/portfolio_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_form_modal.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_model.dart';
import 'package:marshmellow/presentation/widgets/completion_message/completion_message.dart';
import 'package:marshmellow/presentation/pages/cookie/widgets/portfolio/portfolio_edit_modal.dart';
import 'package:marshmellow/presentation/viewmodels/portfolio/portfolio_category_viewmodel.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_category_model.dart';
import 'package:lottie/lottie.dart';

class PortfolioDetailPage extends ConsumerStatefulWidget {
  final int categoryId;

  const PortfolioDetailPage({
    Key? key,
    required this.categoryId,
  }) : super(key: key);

  @override
  ConsumerState<PortfolioDetailPage> createState() =>
      _PortfolioDetailPageState();
}

class _PortfolioDetailPageState extends ConsumerState<PortfolioDetailPage> {
  PortfolioCategoryModel? _category;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      // 모든 포트폴리오 데이터 로드
      await ref.read(portfolioViewModelProvider.notifier).loadPortfolios();

      // 카테고리 정보 로드
      await ref
          .read(portfolioCategoryViewModelProvider.notifier)
          .loadCategories();

      // 현재 카테고리 정보 가져오기
      final categories =
          ref.read(portfolioCategoryViewModelProvider).categories;
      setState(() {
        _category = categories.firstWhere(
          (category) => category.portfolioCategoryPk == widget.categoryId,
          orElse: () => PortfolioCategoryModel(
            portfolioCategoryPk: 0,
            portfolioCategoryName: '카테고리 없음',
            portfolioCategoryMemo: '',
          ),
        );
      });
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
        _loadData(); // 데이터 다시 로드
      } else {
        CompletionMessage.show(context, message: '삭제 실패');
      }
    } catch (e) {
      if (!mounted) return;
      CompletionMessage.show(context, message: '오류 발생');
    }
  }

  @override
  Widget build(BuildContext context) {
    final portfolioState = ref.watch(portfolioViewModelProvider);
    final isLoading = portfolioState.isLoading;
    final errorMessage = portfolioState.errorMessage;

    // 현재 카테고리에 속하는 포트폴리오만 필터링
    final categoryPortfolios = portfolioState.portfolios
        .where((portfolio) =>
            portfolio.portfolioCategory.portfolioCategoryPk ==
            widget.categoryId)
        .toList();

    return Scaffold(
      appBar: CustomAppbar(
        title: _category?.portfolioCategoryName ?? '카테고리',
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () async {
              // 현재 카테고리에 포트폴리오 추가
              final result = await showModalBottomSheet<bool>(
                context: context,
                backgroundColor: Colors.transparent,
                isScrollControlled: true,
                builder: (BuildContext context) {
                  return Modal(
                    backgroundColor: AppColors.background,
                    padding: const EdgeInsets.symmetric(
                        vertical: 20, horizontal: 16),
                    child: PortfolioForm(
                      initialCategoryPk: _category?.portfolioCategoryPk,
                      initialCategoryName: _category?.portfolioCategoryName,
                    ),
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
      body: Stack(
        children: [
          if (isLoading)
            Center(
              child: Lottie.asset(
                'assets/images/loading/loading_simple.json',
                width: 140,
                height: 140,
                fit: BoxFit.contain,
              ),
            )
          else if (errorMessage != null)
            Center(
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
            )
          else if (categoryPortfolios.isEmpty)
            Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Image.asset(
                    'assets/images/characters/char_melong.png',
                    width: 120,
                    height: 120,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    '이 카테고리에는 포트폴리오가 없습니다.\n우측 상단의 + 버튼을 눌러 추가해보세요.',
                    style: AppTextStyles.bodyMedium,
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            )
          else
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // 카테고리 설명 (있을 경우)
                  if (_category != null &&
                      _category!.portfolioCategoryMemo.isNotEmpty) ...[
                    Text(
                      '${_category!.portfolioCategoryMemo}',
                      style: AppTextStyles.bodyMedium.copyWith(
                        color: AppColors.textSecondary,
                      ),
                    ),
                    const SizedBox(height: 16),
                  ],

                  // 포트폴리오 목록
                  Expanded(
                    child: ListView.builder(
                      itemCount: categoryPortfolios.length,
                      itemBuilder: (context, index) {
                        final portfolio = categoryPortfolios[index];
                        return PortfolioItem(
                          portfolio: portfolio,
                          onTap: () => _showPortfolioDetail(portfolio),
                          onDelete: _deletePortfolio,
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
