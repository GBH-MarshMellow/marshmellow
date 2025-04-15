// lib/presentation/pages/finance/demand_detail_page.dart

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:intl/intl.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/finance/detail/demand_detail_model.dart';
import 'package:marshmellow/presentation/pages/finance/services/transfer_service.dart';
import 'package:marshmellow/presentation/viewmodels/finance/demand_detail_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/finance/bank_icon.dart';

class DemandDetailPage extends ConsumerStatefulWidget {
  final String accountNo;
  final String bankName;
  final String accountName;
  final int balance;
  final bool noMoneyMan;

  const DemandDetailPage({
    Key? key,
    required this.accountNo,
    required this.bankName,
    required this.accountName,
    required this.balance,
    required this.noMoneyMan,
  }) : super(key: key);

  @override
  ConsumerState<DemandDetailPage> createState() => _DemandDetailPageState();
}

class _DemandDetailPageState extends ConsumerState<DemandDetailPage> {
  late String startDate;
  late String endDate;
  late DemandDetailParams params;

  @override
  void initState() {
    super.initState();
    // 기본 조회 기간: 최근 1개월
    final now = DateTime.now();
    final oneMonthAgo = DateTime(now.year, now.month - 1, now.day);

    startDate = DateFormat('yyyyMMdd').format(oneMonthAgo);
    endDate = DateFormat('yyyyMMdd').format(now);

    _updateParams();
  }

  // 파라미터 갱신 메서드
  void _updateParams() {
    params = DemandDetailParams(
      accountNo: widget.accountNo,
      startDate: startDate,
      endDate: endDate,
      transactionType: ref.read(transactionFilterProvider),
      orderByType: ref.read(orderByTypeProvider),
    );
  }

  // 상태 변경시 파라미터 갱신
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final currentTransactionType = ref.watch(transactionFilterProvider);
    final currentOrderByType = ref.watch(orderByTypeProvider);

    if (currentTransactionType != params.transactionType ||
        currentOrderByType != params.orderByType) {
      setState(() {
        _updateParams();
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final transactionType = ref.watch(transactionFilterProvider);
    final orderByType = ref.watch(orderByTypeProvider);

    final params = DemandDetailParams(
      accountNo: widget.accountNo,
      startDate: startDate,
      endDate: endDate,
      transactionType: transactionType,
      orderByType: orderByType,
    );

    final transactionsAsync = ref.watch(demandTransactionsProvider(params));

    return Scaffold(
      appBar: CustomAppbar(
        title: 'my little 자산',
        // actions: [
        //             IconButton(
        //     icon: const Icon(Icons.refresh),
        //     color: AppColors.backgroundBlack,
        //     onPressed: () {
        //       ref.read(financeViewModelProvider.notifier).refreshAssetInfo();
        //     },
        //     tooltip: '새로고침',
        //   ),
        // ],
      ),
      body: Column(
        children: [
          _buildAccountHeader(),
          _buildFilterButtons(),
          _buildDateSelector(),
          Expanded(
            child: transactionsAsync.when(
              data: (response) =>
                  _buildTransactionList(response.data?.transactionList ?? []),
              loading: () => Center(
                child: Lottie.asset(
                  'assets/images/loading/loading_simple.json',
                  width: 140,
                  height: 140,
                  fit: BoxFit.contain,
                ),
              ),
              error: (error, stack) =>
                  Center(child: Text('오류가 발생했습니다: $error')),
            ),
          ),
        ],
      ),
    );
  }

  //헤더
  Widget _buildAccountHeader() {
    return Container(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(widget.accountName, style: AppTextStyles.modalTitle),
          SizedBox(
            height: 4,
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.start,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // 은행아이콘
              // BankIcon(bankName: widget.bankName, size: 30),
              // 은행명
              Text(widget.bankName,
                  style: AppTextStyles.bodySmall
                      .copyWith(color: AppColors.divider)),
              const SizedBox(
                width: 4,
              ),
              //계좌번호
              Text(widget.accountNo,
                  style: AppTextStyles.bodySmall
                      .copyWith(color: AppColors.divider)),
              //복사버튼
              InkWell(
                onTap: () {
                  Clipboard.setData(ClipboardData(text: widget.accountNo));
                },
                borderRadius: BorderRadius.circular(4), // 원하는 모서리 둥글기
                child: Padding(
                  padding: const EdgeInsets.all(4), // 필요한 만큼만 패딩 추가
                  child: SvgPicture.asset(
                    'assets/icons/body/CopySimple.svg',
                    height: 16,
                    color: AppColors.disabled,
                  ),
                ),
              )
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('${NumberFormat('#,###').format(widget.balance)}원',
                  style: AppTextStyles.bodyExtraLarge),
              Button(
                text: '송금',
                width: 60,
                height: 40,
                onPressed: () {
                  TransferService.handleTransfer(
                      context, ref, widget.accountNo, widget.bankName);
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  // 전체 입금 출금 필터
  Widget _buildFilterButtons() {
    final transactionType = ref.watch(transactionFilterProvider);
    final viewModel = ref.read(demandDetailViewModelProvider);

    // 드롭다운 옵션 매핑
    final Map<String, String> filterOptions = {'A': '전체', 'M': '입금', 'D': '출금'};

    // 현재 선택된 필터 텍스트
    final String currentFilterText = filterOptions[transactionType] ?? '전체';

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // PopupMenuButton을 Material로 감싸서 너비 설정
          Text(
            '이용내역',
            style: AppTextStyles.bodyMediumLight,
          ),
          Material(
            color: Colors.transparent,
            child: PopupMenuButton<String>(
              padding: EdgeInsets.zero,
              offset: const Offset(0, 20),
              constraints: const BoxConstraints(
                minWidth: 60, // 최소 너비
                maxWidth: 60, // 최대 너비
              ),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(currentFilterText),
                  const Icon(Icons.arrow_drop_down, size: 20),
                ],
              ),
              itemBuilder: (context) => filterOptions.entries.map((entry) {
                return PopupMenuItem<String>(
                  value: entry.key,
                  height: 36, // 작은 높이값
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 4), // 작은 패딩
                  child: Text(entry.value),
                );
              }).toList(),
              onSelected: (String value) {
                viewModel.changeTransactionFilter(value);
              },
            ),
          ),
        ],
      ),
    );
  }

  // 데이트피커
  Widget _buildDateSelector() {
    // 기간 선택 UI 구현 (간단한 버튼 형태로 구현)
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          Expanded(
            child: GestureDetector(
              onTap: () => _selectDate(context, true),
              child: Container(
                padding: const EdgeInsets.all(8),
                child: Text(
                  _formatDateForDisplay(startDate),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 8),
            child: Text('~'),
          ),
          Expanded(
            child: GestureDetector(
              onTap: () => _selectDate(context, false),
              child: Container(
                padding: const EdgeInsets.all(8),
                child: Text(
                  _formatDateForDisplay(endDate),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // 데이트피커 2
  Future<void> _selectDate(BuildContext context, bool isStartDate) async {
    final initialDate = _parseDate(isStartDate ? startDate : endDate);
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: initialDate,
      firstDate: DateTime(2020),
      lastDate: DateTime.now(),
    );

    if (picked != null) {
      setState(() {
        final formattedDate = DateFormat('yyyyMMdd').format(picked);
        if (isStartDate) {
          startDate = formattedDate;
        } else {
          endDate = formattedDate;
        }
        _updateParams(); // 날짜 변경 후 파라미터 갱신
      });
    }
  }

  // 데이트피커 3
  DateTime _parseDate(String dateStr) {
    final year = int.parse(dateStr.substring(0, 4));
    final month = int.parse(dateStr.substring(4, 6));
    final day = int.parse(dateStr.substring(6, 8));
    return DateTime(year, month, day);
  }

  // 데이트피커 4
  String _formatDateForDisplay(String dateStr) {
    final year = dateStr.substring(0, 4);
    final month = dateStr.substring(4, 6);
    final day = dateStr.substring(6, 8);
    return '$year.$month.$day';
  }

  // 내역
  Widget _buildTransactionList(List<TransactionItem> transactions) {
    if (transactions.isEmpty) {
      return const Center(
        child: Text('거래 내역이 없습니다.'),
      );
    }

    // 날짜별로 거래 내역 그룹화
    final Map<String, List<TransactionItem>> groupedTransactions = {};

    for (var item in transactions) {
      final date = item.transactionDate;
      if (!groupedTransactions.containsKey(date)) {
        groupedTransactions[date] = [];
      }
      groupedTransactions[date]!.add(item);
    }

    // 날짜 기준으로 정렬 (최신순)
    final sortedDates = groupedTransactions.keys.toList()
      ..sort((a, b) => b.compareTo(a));

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: sortedDates.length,
      itemBuilder: (context, dateIndex) {
        final date = sortedDates[dateIndex];
        final dateItems = groupedTransactions[date]!;

        // 같은 날짜의 항목들을 시간순으로 정렬
        dateItems
            .sort((a, b) => b.transactionTime.compareTo(a.transactionTime));

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 날짜 헤더
            Padding(
              padding: const EdgeInsets.only(top: 8, bottom: 6),
              child: Text(
                _formatTransactionDate(date),
                style: AppTextStyles.bodyMedium.copyWith(color: Colors.grey),
              ),
            ),
            Divider(
              color: Colors.grey[300],
              thickness: 1.0,
              height: 1.0,
            ),
            SizedBox(
              height: 10,
            ),
            // 해당 날짜의 거래 항목들
            ...dateItems.map((item) {
              final isDeposit = item.transactionType == '1';

              return Padding(
                padding: const EdgeInsets.only(bottom: 20), // 내역간 간격
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 거래 내용
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            item.transactionSummary,
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w400,
                              color: AppColors.textPrimary,
                            ),
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                          ),
                          Row(
                            children: [
                              Text(
                                _formatTransactionTime(item.transactionTime),
                                style: AppTextStyles.bodySmall
                                    .copyWith(color: Colors.grey),
                              ),
                              Text(' | ',
                                  style: AppTextStyles.bodySmall
                                      .copyWith(color: Colors.grey)),
                              Text(
                                isDeposit ? '입금' : '출금',
                                style: AppTextStyles.bodySmall
                                    .copyWith(color: Colors.grey),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),

                    // 금액 및 잔액
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          '${isDeposit ? '+' : '-'}${NumberFormat('#,###').format(int.parse(item.transactionBalance))}원',
                          style: AppTextStyles.bodyMedium.copyWith(
                            fontWeight: FontWeight.bold,
                            color: isDeposit
                                ? AppColors.blueDark
                                : AppColors.warnningLight,
                          ),
                        ),
                        Text(
                          '잔액: ${NumberFormat('#,###').format(int.parse(item.transactionAfterBalance))}원',
                          style: AppTextStyles.bodySmall
                              .copyWith(color: Colors.grey),
                        ),
                      ],
                    ),
                  ],
                ),
              );
            }).toList(),

            // 날짜 그룹 사이 여백
            SizedBox(height: dateIndex < sortedDates.length - 1 ? 8 : 0),
          ],
        );
      },
    );
  }

  // 내역2
  String _formatTransactionDate(String dateStr) {
    final year = int.parse(dateStr.substring(0, 4));
    final month = int.parse(dateStr.substring(4, 6));
    final day = int.parse(dateStr.substring(6, 8));

    final date = DateTime(year, month, day);
    final weekdayName = _getWeekdayName(date.weekday);

    return '$month월 $day일 $weekdayName';
  }

  //내역3
  String _getWeekdayName(int weekday) {
    switch (weekday) {
      case 1:
        return '월요일';
      case 2:
        return '화요일';
      case 3:
        return '수요일';
      case 4:
        return '목요일';
      case 5:
        return '금요일';
      case 6:
        return '토요일';
      case 7:
        return '일요일';
      default:
        return '';
    }
  }

  //내역4
  String _formatTransactionTime(String timeStr) {
    final hour = timeStr.substring(0, 2);
    final minute = timeStr.substring(2, 4);
    return '$hour:$minute';
  }
}
