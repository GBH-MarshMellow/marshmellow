// presentation/pages/finance/finance_page.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/utils/lifecycle/app_lifecycle_manager.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/TopTriangleBubbleWidget.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/finance_analytics_widget.dart';
import 'package:marshmellow/presentation/viewmodels/encryption/encryption_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/loading/loading_manager.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';

// 분리한 위젯들 import
import 'package:marshmellow/presentation/pages/finance/widgets/finance_section_tabs_widget.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/account_item_widget.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/card_item_widget.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/financial_section_widget.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/total_assets_widget.dart';

// appbar 간편버튼
import 'package:marshmellow/presentation/pages/finance/widgets/simple_toggle_button_widget.dart';

// ConsumerStatefulWidget으로 변경
class FinancePage extends ConsumerStatefulWidget {
  const FinancePage({super.key});

  @override
  ConsumerState<FinancePage> createState() => _FinancePageState();
}

class _FinancePageState extends ConsumerState<FinancePage> {
  // 스크롤 컨트롤러를 state에서 관리
  late ScrollController scrollController;

  // 각 섹션의 위치를 참조하기 위한 GlobalKey 맵
  late Map<String, GlobalKey> sectionKeys;

  // 자산 유형분석 위젯 on off
  bool _showAnalyticsWidget = true;

  @override
  void initState() {
    super.initState();

    // 스크롤 컨트롤러 초기화
    scrollController = ScrollController();

    // 섹션 키 초기화
    sectionKeys = {
      '입출금': GlobalKey(),
      '카드': GlobalKey(),
      '예적금': GlobalKey(),
      '대출': GlobalKey(),
    };

    // 위젯 초기화 후 데이터 로드 요청
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(financeViewModelProvider.notifier).fetchAssetInfo();
    });
  }

  @override
  void dispose() {
    // 스크롤 컨트롤러 해제
    scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 라이프사이클 상태 구독
    final lifecycleState = ref.watch(lifecycleStateProvider);

    // 뷰모델에서 데이터 가져오기
    final financeState = ref.watch(financeViewModelProvider);

    // 로딩 상태에 따라 LoadingManager 처리
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (financeState.isLoading) {
        LoadingManager.show(context, text: '자산 정보를 불러오는 중...');
      } else {
        LoadingManager.hide();
      }
    });

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: CustomAppbar(
        title: 'my little 자산',
        actions: [
          const SimpleToggleButton(), // 분리한 커스텀 토글 버튼 위젯
          IconButton(
            icon: const Icon(Icons.refresh),
            color: AppColors.backgroundBlack,
            onPressed: () {
              ref.read(financeViewModelProvider.notifier).refreshAssetInfo();
            },
            tooltip: '새로고침',
          ),
        ],
      ),
      body: Center(
        child: Column(
          children: [
            // 데이터 로딩 중이 아닐 때만 탭 표시
            if (financeState.assetData != null && !financeState.isLoading)
              FinanceSectionTabs(
                scrollController: scrollController,
                sectionKeys: sectionKeys,
              ),

            Expanded(
              child: _buildContent(financeState),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildContent(FinanceState state) {
    // 로딩 중
    if (state.isLoading && state.assetData == null) {
      return Container(); // LoadingManager가 오버레이로 표시됨
    }

    // 에러 발생
    if (state.error != null && state.assetData == null) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Lottie.asset(
              'assets/images/loading/secure.json',
              width: 100,
              height: 100,
              fit: BoxFit.contain,
            ),
          const SizedBox(height: 10),
          // Text('에러 발생: ${state.error}'),
          Button(
            width: MediaQuery.of(context).size.width *0.5,
            onPressed: () {
                ref.read(aesKeyNotifierProvider.notifier).fetchAesKey();
                ref.read(financeViewModelProvider.notifier).refreshAssetInfo();},
            text: '보안인증',
          ),
        ],
      );
    }

    // 데이터 없음
    if (state.assetData == null) {
      return const Center(child: Text('자산 데이터가 없습니다'));
    }

    // 성공적으로 데이터를 받았을 때
    final viewModel = ref.read(financeViewModelProvider.notifier);
    final data = state.assetData!;
    final totalAssets = viewModel.calculateTotalAssets();
    final currentMonth = DateTime.now().month;

    return SingleChildScrollView(
      controller: scrollController, // 스크롤 컨트롤러 연결
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 12),
          // 총 자산 정보
          TotalAssetsWidget(totalAssets: totalAssets , scrollController: scrollController),
          // 클릭 버튼
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const TopTriangleBubbleWidget(
                text: 'click',
                baseBackgroundColor: Color.fromARGB(255, 211, 211, 211),
                pulseBackgroundColor: Color.fromARGB(255, 180, 180, 180),
                pulseDuration: Duration(seconds: 1),
                width: 45,
                height: 30,
              ),
            ],
          ),
          // 입출금 계좌 정보
          FinancialSectionWidget(
            key: sectionKeys['입출금'], // 섹션 위치 추적용 키
            title: '입출금',
            // String 타입의 totalAmount를 int로 변환하거나 String 타입 그대로 사용
            totalAmount: int.tryParse(data.data.demandDepositData.totalAmount) ?? 0,
            isEmpty: data.data.demandDepositData.demandDepositList.isEmpty,
            emptyMessage: '등록된 입출금 계좌가 없습니다.',
            itemList: data.data.demandDepositData.demandDepositList
                .map((account) => AccountItemWidget(
                    bankName: account.bankName,
                    accountName: account.accountName,
                    accountNo: account.accountNo,
                    // encodedAccountBalance를 사용하거나 복호화된 accountBalance 사용
                    balance: account.encodedAccountBalance != null
                        ? int.tryParse(account.encodedAccountBalance!) ?? 0
                        : account.accountBalance ?? 0,
                    noMoneyMan: true,
                    type: '입출금'))
                .toList(),
          ),

          // 카드 정보
          FinancialSectionWidget(
            key: sectionKeys['카드'], // 섹션 위치 추적용 키
            title: '${currentMonth}월 카드 결제',
            totalAmount: int.tryParse(data.data.cardData.totalAmount)?? 0,
            isEmpty: data.data.cardData.cardList.isEmpty,
            emptyMessage: '등록된 카드가 없습니다.',
            itemList: data.data.cardData.cardList
                .map((card) => CardItemWidget(card: card))
                .toList(),
          ),

          // 자산 유형
          if (_showAnalyticsWidget)
            FinanceAnalyticsWidget(
              onClose: () {
                setState(() {
                  _showAnalyticsWidget = false;
                });
              },
            ),
          const SizedBox(height: 12),

          // 예금 정보
          FinancialSectionWidget(
            key: sectionKeys['예적금'], // 섹션 위치 추적용 키
            title: '예금',
            totalAmount: int.tryParse(data.data.depositData.totalAmount) ?? 0,
            isEmpty: data.data.depositData.depositList.isEmpty,
            emptyMessage: '등록된 예금이 없습니다.',
            itemList: data.data.depositData.depositList
                .map((account) => AccountItemWidget(
                    bankName: account.bankName,
                    accountName: account.accountName,
                    accountNo: account.accountNo,
                    // encodeDepositBalance를 사용하거나 복호화된 depositBalance 사용
                    balance: account.encodeDepositBalance != null
                        ? int.tryParse(account.encodeDepositBalance!) ?? 0
                        : account.depositBalance ?? 0,
                    type: '예금'))
                .toList(),
          ),

          // 적금 정보
          FinancialSectionWidget(
            title: '적금',
            // String 타입의 totalAmount를 int로 변환
            totalAmount: int.tryParse(data.data.savingsData.totalAmount) ?? 0,
            isEmpty: data.data.savingsData.savingsList.isEmpty,
            emptyMessage: '등록된 적금이 없습니다.',
            itemList: data.data.savingsData.savingsList
                .map((account) => AccountItemWidget(
                    bankName: account.bankName,
                    accountName: account.accountName,
                    accountNo: account.accountNo,
                    // encodedTotalBalance를 사용하거나 복호화된 totalBalance 사용
                    balance: account.encodedTotalBalance != null
                        ? int.tryParse(account.encodedTotalBalance!) ?? 0
                        : account.totalBalance ?? 0,
                    type: '적금'))
                .toList(),
          ),

          // 대출 정보
          FinancialSectionWidget(
            key: sectionKeys['대출'], // 섹션 위치 추적용 키
            title: '대출',
            // String 타입의 totalAmount를 int로 변환
            totalAmount: int.tryParse(data.data.loanData.totalAmount) ?? 0,
            isEmpty: data.data.loanData.loanList.isEmpty,
            emptyMessage: '등록된 대출이 없습니다.',
            itemList: data.data.loanData.loanList
                .map((loan) => AccountItemWidget(
                    bankName: '-', // 은행명 정보 없음
                    accountName: loan.accountName,
                    accountNo: loan.accountNo,
                    // encodeLoanBalance를 사용하거나 복호화된 loanBalance 사용
                    balance: loan.encodeLoanBalance != null
                        ? int.tryParse(loan.encodeLoanBalance!) ?? 0
                        : loan.loanBalance ?? 0,
                    isLoan: true,
                    type: '대출'))
                .toList(),
          ),

        ],
      ),
    );
  }
}

