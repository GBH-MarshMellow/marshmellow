import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/constants/icon_path.dart';

import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/custom_search_bar/custom_search_bar.dart';
import 'package:marshmellow/core/services/hive_service.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/search/no_recent_searches.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/search/recent_searches.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_item.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/search_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/router/routes/ledger_routes.dart';
import 'package:go_router/go_router.dart';

class LedgerSearchPage extends ConsumerStatefulWidget {
  const LedgerSearchPage({super.key});

  @override
  ConsumerState<LedgerSearchPage> createState() => _LedgerSearchPageState();
}

class _LedgerSearchPageState extends ConsumerState<LedgerSearchPage> {
  final TextEditingController _searchController = TextEditingController();
  List<String> _recentSearches = [];
  final FocusNode _searchFocusNode = FocusNode();

  @override
  void initState() {
    super.initState();
    _loadSearchHistory();

    // 페이지에 들어올 때 검색 결과 초기화
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(searchViewModelProvider.notifier).clearSearch();
      _searchController.clear();

      // 자동으로 검색 입력 필드에 포커스 설정하여 키보드가 나타나도록 함
      FocusScope.of(context).requestFocus(_searchFocusNode);
    });
  }

  @override
  void dispose() {
    // 페이지를 떠날 때 검색 결과 초기화
    ref.read(searchViewModelProvider.notifier).clearSearch();
    _searchController.dispose();
    _searchFocusNode.dispose();
    super.dispose();
  }

  // 검색 히스토리 불러오기
  Future<void> _loadSearchHistory() async {
    final history = await SearchHistoryService.getSearchHistory();
    setState(() {
      _recentSearches = history;
    });
  }

  // 검색어 추가
  void _addSearchTerm(String term) async {
    if (term.trim().isEmpty) return;

    await SearchHistoryService.saveSearch(term);
    await _loadSearchHistory();
  }

  // 검색어 삭제
  void _removeSearchTerm(String term) async {
    await SearchHistoryService.removeSearchTerm(term);
    await _loadSearchHistory();
  }

  // 모든 검색어 삭제
  void _clearAllSearchHistory() async {
    await SearchHistoryService.clearSearchHistory();
    await _loadSearchHistory();
  }

  // 검색 수행
  void _performSearch(String term) {
    if (term.trim().isEmpty) return;

    // 검색어 히스토리에 추가
    _addSearchTerm(term);

    // 시작일은 충분히 과거, 종료일은 오늘 날짜로 설정
    String startDate = '20000101'; // 2000년 1월 1일부터
    String endDate = DateFormat('yyyyMMdd').format(DateTime.now()); // 오늘까지

    // 검색 실행
    ref.read(searchViewModelProvider.notifier).search(
          keyword: term,
          startDate: startDate,
          endDate: endDate,
        );

    // 키보드 닫기
    FocusScope.of(context).unfocus();
  }

  // 검색 결과 지우기
  void _clearSearchResults() {
    ref.read(searchViewModelProvider.notifier).clearSearch();
    setState(() {
      _searchController.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    // 검색 상태 구독
    final searchState = ref.watch(searchViewModelProvider);

    return Scaffold(
      appBar: CustomAppbar(title: '검색', actions: [
        IconButton(
          icon: SvgPicture.asset(IconPath.analysis),
          onPressed: () {
            context.push(LedgerRoutes.getAnalysisPath());
          },
        )
      ]),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: CustomSearchBar(
                controller: _searchController,
                focusNode: _searchFocusNode,
                onChanged: (value) {},
                onSearchPressed: () => _performSearch(_searchController.text),
                onSubmitted: (value) => _performSearch(value),
              ),
            ),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: searchState.searchTerm.isEmpty
                    ? _buildRecentSearches()
                    : _buildSearchResults(searchState),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // 검색 결과 위젯
  Widget _buildSearchResults(SearchState searchState) {
    if (searchState.isLoading) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    if (searchState.errorMessage != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('오류가 발생했습니다: ${searchState.errorMessage}'),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _clearSearchResults,
              child: const Text('다시 시도하기'),
            ),
          ],
        ),
      );
    }

    if (searchState.searchResults.isEmpty) {
      // 결과가 없을 때 표시할 화면
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            SizedBox(height: 60),
            Image.asset(
              'assets/images/characters/char_angry_notebook.png',
              height: 150,
            ),
            const SizedBox(height: 30),
            Text(
              '\'${searchState.searchTerm}\'에 대한 \n검색 결과가 없습니다.',
              style: AppTextStyles.bodyMedium
                  .copyWith(color: AppColors.textPrimary),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    // 트랜잭션을 날짜별로 그룹화
    final repository = ref.watch(transactionRepositoryProvider);
    final transactions = searchState.searchResults;
    final grouped = repository.groupTransactionsByDate(transactions);

    // 날짜 목록을 내림차순으로 정렬 (최신순)
    final dates = grouped.keys.toList()..sort((a, b) => b.compareTo(a));

    // 결과 헤더
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              '"${searchState.searchTerm}" 검색 결과',
              style: AppTextStyles.bodyMedium,
            ),
            TextButton(
              onPressed: _clearSearchResults,
              child: Text(
                '초기화',
                style: AppTextStyles.bodySmall.copyWith(
                  color: AppColors.textSecondary,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),

        // 날짜별 거래 목록
        Expanded(
          child: ListView.builder(
            physics: const AlwaysScrollableScrollPhysics(),
            shrinkWrap: true,
            itemCount: dates.length,
            itemBuilder: (context, index) {
              final date = dates[index];
              final items = grouped[date]!;

              // 요일 이름 (월요일, 화요일, ...)
              final dayNames = [
                '월요일',
                '화요일',
                '수요일',
                '목요일',
                '금요일',
                '토요일',
                '일요일'
              ];
              final dayName = dayNames[date.weekday - 1];

              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // 날짜 헤더
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 5.0),
                    child: Text(
                      '${date.month}월 ${date.day}일 $dayName',
                      style: AppTextStyles.bodyMedium.copyWith(
                        fontWeight: FontWeight.w300,
                        color: AppColors.textPrimary,
                      ),
                    ),
                  ),
                  // 날짜 바로 아래에 Divider 추가
                  const Divider(
                    height: 1,
                    thickness: 0.5,
                    color: AppColors.textSecondary,
                  ),

                  // 해당 날짜의 거래 목록
                  const SizedBox(height: 10),
                  ...items.map((item) => Padding(
                        padding: const EdgeInsets.only(bottom: 12),
                        child: TransactionListItem(
                          transaction: item,
                        ),
                      )),
                  const SizedBox(height: 10),
                ],
              );
            },
          ),
        ),
      ],
    );
  }

  // 최근 검색어 위젯
  Widget _buildRecentSearches() {
    if (_recentSearches.isEmpty) {
      return const NoRecentSearches();
    }

    return RecentSearches(
      searches: _recentSearches,
      onSearchTermSelected: (term) {
        _searchController.text = term;
        _performSearch(term);
      },
      onSearchTermDeleted: _removeSearchTerm,
      onClearAll: _clearAllSearchHistory,
    );
  }
}
