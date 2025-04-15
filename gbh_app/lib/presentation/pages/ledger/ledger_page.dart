import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:flutter/foundation.dart';

// 패키지
import 'package:flutter_svg/flutter_svg.dart';
import 'package:syncfusion_flutter_datepicker/datepicker.dart';

// 라우터
import 'package:marshmellow/router/routes/ledger_routes.dart';
import 'package:go_router/go_router.dart';

// 상태관리
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/ledger_viewmodel.dart';
import 'package:marshmellow/di/providers/date_picker_provider.dart';
import 'package:marshmellow/di/providers/transaction_filter_provider.dart';
import 'package:marshmellow/presentation/viewmodels/ledger/transaction_list_viewmodel.dart';
import 'package:marshmellow/core/services/transaction_classifier_service.dart';

// 위젯
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/main/financial_card.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/main/page_dot_indicator.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/main/ledger_transaction_history.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/main/ledger_calendar.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/main/date_range_selector.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/transaction_modal/transaction_form/transaction_form.dart';
import 'package:marshmellow/presentation/pages/ledger/widgets/picker/filter.dart';

class LedgerPage extends ConsumerStatefulWidget {
  const LedgerPage({super.key});

  @override
  ConsumerState<LedgerPage> createState() => _LedgerPageState();
}

class _LedgerPageState extends ConsumerState<LedgerPage> {
  final PageController _pageController = PageController();
  int _currentPage = 0;
  final GlobalKey _filterDropdownKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    // 앱 시작 시 날짜 범위 설정
    // 빌드 완료 후 실행되도록 설정
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initializeDefaultDateRange();
    });
  }

  // 초기 날짜 범위 설정
  void _initializeDefaultDateRange() {
    final datePickerState = ref.read(datePickerProvider);

    // 날짜 범위가 아직 설정되지 않았으면 현재 월로 설정
    if (datePickerState.selectedRange == null ||
        datePickerState.selectedRange!.startDate == null) {
      final now = DateTime.now();
      final firstDay = DateTime(now.year, now.month, 1);
      final lastDay = DateTime(now.year, now.month + 1, 0);

      // DatePicker 상태 업데이트
      ref
          .read(datePickerProvider.notifier)
          .updateSelectedRange(PickerDateRange(firstDay, lastDay));

      // 데이터 로드
      ref
          .read(ledgerViewModelProvider.notifier)
          .loadHouseholdData(PickerDateRange(firstDay, lastDay));
    } else {
      // 이미 날짜 범위가 있다면 해당 범위로 데이터 로드
      ref
          .read(ledgerViewModelProvider.notifier)
          .loadHouseholdData(datePickerState.selectedRange!);
    }
  }

  @override
  Widget build(BuildContext context) {
    // 화면 크기 정의
    final screenWidth = MediaQuery.of(context).size.width;
    final contentWidth = screenWidth * 0.9;
    final ledgerState = ref.watch(ledgerViewModelProvider);

    // 현재 선택된 필터 상태
    final currentFilter = ref.watch(transactionFilterProvider);

    return Scaffold(
      appBar: CustomAppbar(title: '가계부', actions: [
        IconButton(
          icon: SvgPicture.asset(IconPath.analysis),
          onPressed: () {
            context.push(LedgerRoutes.getAnalysisPath());
          },
        )
      ]),
      body: SafeArea(
        child: RefreshIndicator(
          onRefresh: () async {
            // 트랜잭션 동기화 수행
            final syncService = ref.read(transactionSyncServiceProvider);
            try {
              final hasUnsortedTransactions =
                  await syncService.hasUnsortedTransactions();
              if (hasUnsortedTransactions) {
                await syncService.performFullSync();
              }
            } catch (e) {
              if (kDebugMode) {
                print('❌ 트랜잭션 동기화 중 오류 발생: $e');
              }
            }

            // 데이터 새로고침
            ref.invalidate(transactionsProvider);
            ref.invalidate(filteredTransactionsProvider);

            // 현재 선택된 날짜 범위로 데이터 다시 로드
            final datePickerState = ref.read(datePickerProvider);
            if (datePickerState.selectedRange != null) {
              await ref
                  .read(ledgerViewModelProvider.notifier)
                  .loadHouseholdData(datePickerState.selectedRange!);
            }
          },
          child: SingleChildScrollView(
            physics: const AlwaysScrollableScrollPhysics(),
            child: Center(
              child: Container(
                width: contentWidth,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 날짜 선택 컴포넌트
                    DateRangeSelector(
                      onPreviousPressed: () {
                        print('이전 기간으로 이동했습니다');
                      },
                      onNextPressed: () {
                        print('다음 기간으로 이동했습니다');
                      },
                    ),

                    // 수입/지출 카드 영역
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: FinanceCard(
                            title: '수입',
                            amount: ledgerState.totalIncome,
                            backgroundColor: AppColors.bluePrimary,
                          ),
                        ),
                        SizedBox(width: screenWidth * 0.03),
                        Expanded(
                          child: FinanceCard(
                            title: '지출',
                            amount: ledgerState.totalExpenditure,
                            backgroundColor: AppColors.pinkPrimary,
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Row(
                          children: [
                            Text('필터',
                                style: AppTextStyles.bodyMedium
                                    .copyWith(fontWeight: FontWeight.w300)),
                            SizedBox(width: screenWidth * 0.03),
                            GestureDetector(
                              key: _filterDropdownKey,
                              onTap: () {
                                context.showTransactionFilterDropdown(
                                  dropdownKey: _filterDropdownKey,
                                  onFilterSelected: (filter) {
                                    print('선택된 필터: $filter');
                                    ref
                                        .read(
                                            transactionFilterProvider.notifier)
                                        .state = filter;
                                  },
                                );
                              },
                              child: SvgPicture.asset(IconPath.caretDown),
                            ),
                          ],
                        ),
                        Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            GestureDetector(
                              onTap: () {
                                // plus 버튼 액션
                                showCustomModal(
                                  context: context,
                                  ref: ref,
                                  backgroundColor: AppColors.background,
                                  child: const TransactionForm(),
                                );
                              },
                              child: Padding(
                                padding: EdgeInsets.only(right: 10),
                                child: SvgPicture.asset(
                                  IconPath.plus,
                                  width: 23,
                                  height: 23,
                                ),
                              ),
                            ),
                            GestureDetector(
                              onTap: () {
                                // 검색 페이지로 이동
                                context.push(LedgerRoutes.getSearchPath());
                              },
                              child: SvgPicture.asset(
                                IconPath.searchOutlined,
                                width: 23,
                                height: 23,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),

                    const SizedBox(height: 16),

                    // 페이지 인디케이터
                    Center(
                      child: PageDotIndicator(
                        currentPage: _currentPage,
                        totalPages: 2,
                        pageController: _pageController,
                        onPageChanged: (index) {
                          setState(() {
                            _currentPage = index;
                          });
                        },
                      ),
                    ),

                    // 페이지 뷰 컨테이너
                    SizedBox(
                      height: 450, // 필요한 높이로 조정
                      child: PageView(
                        controller: _pageController,
                        onPageChanged: (index) {
                          setState(() {
                            _currentPage = index;
                          });
                        },
                        children: const [
                          // 첫 번째 페이지 - 거래 내역
                          LedgerTransactionHistory(),

                          // 두 번째 페이지 - 캘린더
                          LedgerCalendar(),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }
}
