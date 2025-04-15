// lib/presentation/pages/finance/card_detail_page.dart

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/finance/detail/card_detail_model.dart';
import 'package:marshmellow/presentation/viewmodels/finance/card_detail_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/finance/card_image_util.dart';

class CardDetailPage extends ConsumerStatefulWidget {
  final String cardNo;
  final String bankName;
  final String cardName;
  final String cvc;
  final int balance;

  const CardDetailPage({
    Key? key,
    required this.cardNo,
    required this.bankName,
    required this.cardName,
    required this.cvc,
    required this.balance,
  }) : super(key: key);

  @override
  ConsumerState<CardDetailPage> createState() => _CardDetailPageState();
}

class _CardDetailPageState extends ConsumerState<CardDetailPage> {
  late String startDate;
  late String endDate;
  late CardDetailParams params;

  @override
  void initState() {
    super.initState();
    // 기본 조회 기간: 최근 1개월
    final now = DateTime.now();
    final oneMonthAgo = DateTime(now.year, now.month - 1, now.day);

    startDate = DateFormat('yyyyMMdd').format(oneMonthAgo);
    endDate = DateFormat('yyyyMMdd').format(now);

    // 초기 파라미터 설정
    _updateParams();
  }

  // 파라미터 갱신 메서드
  void _updateParams() {
    params = CardDetailParams(
      cardNo: widget.cardNo,
      cvc: widget.cvc,
      startDate: startDate,
      endDate: endDate,
    );
  }

  @override
  Widget build(BuildContext context) {
    final transactionsAsync = ref.watch(cardTransactionsProvider(params));

    return Scaffold(
      appBar: CustomAppbar(
        title: 'my little 자산',
      ),
      body: Column(
        children: [
          _buildCardHeader(),
          Row(
            children: [
              Expanded(child: _buildDateSelector()),
              Container(
                // color: Colors.amber,
                child: transactionsAsync.when(
                  data: (response) => _buildEstimatedBalance(
                      int.tryParse(response.data.estimatedBalance) ?? 0),
                  loading: () => const SizedBox(
                    height: 80, // _buildDateSelector와 비슷한 높이로 설정
                  ),
                  error: (_, __) => const SizedBox(
                    height: 80, // _buildDateSelector와 비슷한 높이로 설정
                  ),
                ),
              ),
            ],
          ),
          Expanded(
            child: transactionsAsync.when(
              data: (response) =>
                  _buildTransactionList(response.data.transactionList),
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

  Widget _buildCardHeader() {
    return Container(
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.all(20),
      // color: Colors.white,
      decoration: BoxDecoration(
          border: Border.all(width: 1.0, color: AppColors.divider),
          borderRadius: BorderRadius.all(Radius.circular(10.0))),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(widget.cardName, style: AppTextStyles.bodyLarge),
              const SizedBox(height: 4),
              Text('신용 | ${_formatCardNumber(widget.cardNo)}',
                  style: AppTextStyles.bodyMedium
                      .copyWith(color: AppColors.divider)),
              const SizedBox(height: 16),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('결제 예정 금액',
                      style: AppTextStyles.bodyMedium
                          .copyWith(color: AppColors.divider)),
                  Text('${NumberFormat('#,###').format(widget.balance)}원',
                      style: AppTextStyles.bodyExtraLarge),
                ],
              ),
            ],
          ),
          CardImageUtil.getCardImageWidget(widget.cardName, size: 128),
        ],
      ),
    );
  }

  // 카드번호 형식화 (1234-5678-9012-3456)
  String _formatCardNumber(String cardNo) {
    if (cardNo.length != 16) return cardNo;
    // return '${cardNo.substring(0, 4)}-${cardNo.substring(4, 8)}-${cardNo.substring(8, 12)}-${cardNo.substring(12, 16)}';
    return cardNo.substring(12, 16);
  }

  Widget _buildDateSelector() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('조회 기간', style: AppTextStyles.bodyMedium),
          const SizedBox(height: 4),
          GestureDetector(
            onTap: () => _selectDate(context, true),
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.calendar_today,
                      size: 16, color: AppColors.divider),
                  const SizedBox(width: 4),
                  Text(
                    '시작: ${_formatDateForDisplay(startDate)}',
                    style: AppTextStyles.bodySmall
                        .copyWith(color: AppColors.textSecondary),
                  ),
                ],
              ),
            ),
          ),
          GestureDetector(
            onTap: () => _selectDate(context, false),
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.calendar_today,
                      size: 16, color: AppColors.divider),
                  const SizedBox(width: 4),
                  Text(
                    '종료: ${_formatDateForDisplay(endDate)}',
                    style: AppTextStyles.bodySmall
                        .copyWith(color: AppColors.textSecondary),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

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

  DateTime _parseDate(String dateStr) {
    final year = int.parse(dateStr.substring(0, 4));
    final month = int.parse(dateStr.substring(4, 6));
    final day = int.parse(dateStr.substring(6, 8));
    return DateTime(year, month, day);
  }

  String _formatDateForDisplay(String dateStr) {
    final year = dateStr.substring(0, 4);
    final month = dateStr.substring(4, 6);
    final day = dateStr.substring(6, 8);
    return '$year.$month.$day';
  }

  Widget _buildEstimatedBalance(int estimatedBalance) {
    return Container(
      height: 80, // _buildDateSelector와 비슷한 높이로 맞춤
      padding: const EdgeInsets.fromLTRB(0, 4.0, 16.0, 0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.start, // 세로 중앙 정렬 추가
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text('총 사용금액', style: AppTextStyles.bodyMedium),
          const SizedBox(height: 16), // 간격 약간 늘림
          Text(
            '${NumberFormat('#,###').format(estimatedBalance)}원',
            style: AppTextStyles.bodyLarge.copyWith(
              fontWeight: FontWeight.bold,
              color: AppColors.blueDark,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTransactionList(List<CardTransactionItem> transactions) {
    if (transactions.isEmpty) {
      return const Center(
        child: Text('거래 내역이 없습니다.'),
      );
    }

    // 날짜별로 거래 내역 그룹화
    final Map<String, List<CardTransactionItem>> groupedTransactions = {};

    for (var item in transactions) {
      // ****************************************
      // 수정: null 체크 추가 - transactionDate가 null이면 '알 수 없음'으로 처리
      // ****************************************
      final date = item.transactionDate ?? '알 수 없음';
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
        // ****************************************
        // 수정: null 체크 추가 - transactionTime이 null인 경우 빈 문자열로 처리
        // ****************************************
        dateItems.sort((a, b) =>
            (b.transactionTime ?? '').compareTo(a.transactionTime ?? ''));

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 날짜 헤더
            Padding(
              padding: const EdgeInsets.only(top: 16),
              child: Text(
                _formatTransactionDate(date),
                style: AppTextStyles.bodyMedium.copyWith(color: Colors.grey),
              ),
            ),
            Divider(
              color: AppColors.disabled,
            ),
            SizedBox(
              height: 16,
            ),

            // 해당 날짜의 거래 항목들
            ...dateItems.map((item) {
              // ****************************************
              // 수정: null 체크 추가 - cardStatus가 null인 경우 기본값 제공
              // ****************************************
              final isApproved = (item.cardStatus ?? '') == '승인';

              return Padding(
                padding: const EdgeInsets.only(bottom: 22),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 거래 내용
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            // ****************************************
                            // 수정: null 체크 추가 - merchantName이 null인 경우 '상호명 없음' 표시
                            // ****************************************
                            item.merchantName ?? '상호명 없음',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w400,
                              color: AppColors.textPrimary,
                            ),
                            overflow: TextOverflow.ellipsis,
                          ),
                          // SizedBox(height: 4,),
                          Row(
                            children: [
                              SizedBox(
                                // width: 50,
                                child: Text(
                                  // ****************************************
                                  // 수정: null 체크 추가 - transactionTime이 null인 경우 빈 문자열로 처리
                                  // ****************************************
                                  _formatTransactionTime(
                                      item.transactionTime ?? ''),
                                  style: AppTextStyles.bodySmall
                                      .copyWith(color: Colors.grey),
                                ),
                              ),
                              Text(
                                ' | ',
                                style: AppTextStyles.bodySmall
                                    .copyWith(color: Colors.grey),
                              ),
                              Text(
                                // ****************************************
                                // 수정: null 체크 추가 - categoryName이 null인 경우 '분류 없음' 표시
                                // ****************************************
                                item.categoryName ?? '분류 없음',
                                style: AppTextStyles.bodySmall
                                    .copyWith(color: Colors.grey),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),

                    // 금액 및 상태
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          // ****************************************
                          // 수정: null 체크 및 예외 처리 추가 - transactionBalance가 null이거나 파싱 실패 시 '0'으로 처리
                          // ****************************************
                          '${NumberFormat('#,###').format(int.tryParse(item.transactionBalance ?? '0') ?? 0)}원',
                          style: AppTextStyles.bodyMedium.copyWith(
                            fontWeight: FontWeight.bold,
                            color: isApproved
                                ? AppColors.blueDark
                                : AppColors.warnningLight,
                          ),
                        ),
                        Row(
                          children: [
                            Text(
                              // ****************************************
                              // 수정: null 체크 추가 - billStatementsStatus가 null인 경우 빈 문자열로 처리
                              // ****************************************
                              item.billStatementsStatus ?? '',
                              style: AppTextStyles.bodySmall
                                  .copyWith(color: Colors.grey),
                            ),
                            const SizedBox(width: 4),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 6, vertical: 2),
                              decoration: BoxDecoration(
                                color: isApproved
                                    ? AppColors.blueDark.withOpacity(0.1)
                                    : AppColors.warnningLight.withOpacity(0.1),
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: Text(
                                // ****************************************
                                // 수정: null 체크 추가 - cardStatus가 null인 경우 '상태 없음' 표시
                                // ****************************************
                                item.cardStatus ?? '상태 없음',
                                style: AppTextStyles.bodySmall.copyWith(
                                  color: isApproved
                                      ? AppColors.blueDark
                                      : AppColors.warnningLight,
                                ),
                              ),
                            ),
                          ],
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

  String _formatTransactionDate(String dateStr) {
    final year = int.parse(dateStr.substring(0, 4));
    final month = int.parse(dateStr.substring(4, 6));
    final day = int.parse(dateStr.substring(6, 8));

    final date = DateTime(year, month, day);
    final weekdayName = _getWeekdayName(date.weekday);

    return '$month월 $day일 $weekdayName';
  }

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

  String _formatTransactionTime(String timeStr) {
    final hour = timeStr.substring(0, 2);
    final minute = timeStr.substring(2, 4);
    return '$hour:$minute';
  }
}
