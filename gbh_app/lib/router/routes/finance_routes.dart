// lib/router/routes/finance_routes.dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/presentation/pages/finance/certificate_auth_page.dart';
import 'package:marshmellow/presentation/pages/finance/detail/card_detail_page.dart';
import 'package:marshmellow/presentation/pages/finance/detail/demand_detail_page.dart';
import 'package:marshmellow/presentation/pages/finance/detail/deposit_detail_page.dart';
import 'package:marshmellow/presentation/pages/finance/detail/loan_detail_page.dart';
import 'package:marshmellow/presentation/pages/finance/detail/saving_detail_page.dart';
import 'package:marshmellow/presentation/pages/finance/finance_agreement_detail_page.dart';
import 'package:marshmellow/presentation/pages/finance/finance_analysis_page.dart';
import 'package:marshmellow/presentation/pages/finance/finance_page.dart';
import 'package:marshmellow/presentation/pages/finance/finance_test_page.dart';
import 'package:marshmellow/presentation/pages/finance/finance_transfer_page.dart';
import 'package:marshmellow/presentation/pages/finance/simple_finance_page.dart';
import 'package:marshmellow/presentation/pages/finance/transfer_complete_page.dart';
import 'package:marshmellow/presentation/pages/finance/transfer_page.dart';
import 'package:marshmellow/presentation/pages/finance/withdrawal_account_registration_page.dart';
import 'package:marshmellow/presentation/pages/testpage/keyboard_test_page.dart';

class FinanceRoutes {
  static const String root = '/finance';
  static const String test = 'financetest'; // 테스트페이지
  static const String keyboardtest = 'keyboardtest'; // 키보드테스트
  static const String rootsimple = 'simple'; // 간편페이지 
  static const String transfer = 'transfer'; // 송금페이지
  static const String analysis = 'analysis'; // 자산유형분석
  static const String demandDetail = 'account/demand/:accountNo'; // 입출금계좌 상세 경로
  static const String depositDetail = 'account/deposit/:accountNo'; // 예금계좌 상세 경로
  static const String savingDetail = 'account/saving/:accountNo'; // 적금계좌 상세 경로
  static const String loanDetail = 'account/loan/:accountNo'; // 대출계좌 상세 경로
  static const String cardDetail = 'card/:cardNo'; // 카드계좌 상세 경로
  static const String withdrawalAccountRegistration = 'account/withdrawal-registration/:accountNo'; // 출금계좌 등록
  static const String auth = 'auth'; // 송금전 인증페이지
  static const String agreement = 'agreement/:agreementNo'; // 약관동의 상세
  static const String transferComplete = 'transfer-complete'; // 송금완료 페이지
  
  
  // 전체 경로 생성 헬퍼 메서드
  static String getTestPath() => '$root/$test'; // 테스트페이지
  static String getKeyboardTestPath() => '$root/$keyboardtest'; // 키보드테스트
  static String getSimplePath() => '$root/$rootsimple'; // 간편페이지
  static String getTransferPath() => '$root/$transfer'; // 송금페이지
  static String getAnalysisPath() => '$root/$analysis'; // 자산유형분석
  static String getDemandDetailPath(String accountNo) => '$root/account/demand/$accountNo'; // 입출금계좌 상세 경로
  static String getDepositDetailPath(String accountNo) => '$root/account/deposit/$accountNo'; // 예금계좌 상세 경로
  static String getSavingDetailPath(String accountNo) => '$root/account/saving/$accountNo'; // 적금계좌 상세 경로
  static String getLoanDetailPath(String accountNo) => '$root/account/loan/$accountNo'; // 대출계좌 상세 경로
  static String getCardDetailPath(String cardNo) => '$root/card/$cardNo'; // 카드계좌 상세 경로
  static String getWithdrawalAccountRegistrationPath(String accountNo) => '$root/account/withdrawal-registration/$accountNo'; // 출금계좌 등록
  static String getAuthPath() => '$root/$auth'; // 송금전 인증페이지 
  static String getAgreementPath(String agreementNo) => '$root/agreement/$agreementNo'; // 약관동의 상세
  static String getTransferCompletePath() => '$root/$transferComplete'; // 송금완료 페이지 경로
}

List<RouteBase> financeRoutes = [
  GoRoute(
    path: FinanceRoutes.root,
    builder: (context, state) => const FinancePage(),
    routes: [
      // 테스트 하위 라우트 추가
      GoRoute(
        path: FinanceRoutes.test,
        builder: (context, state) => const FinanceTestPage(),
      ),
      GoRoute(
        path: FinanceRoutes.keyboardtest,
        builder: (context, state) => const KeyboardTestPage(),
      ),
      GoRoute(
        path: FinanceRoutes.rootsimple,
        builder: (context, state) => const SimpleFinancePage(),
      ),
      // 송금 페이지 라우트 수정
GoRoute(
  path: FinanceRoutes.transfer,
  builder: (context, state) {
    final extra = state.extra as Map<String, dynamic>?;
    final accountNo = extra?['accountNo'] as String? ?? '';
    final withdrawalAccountId = extra?['withdrawalAccountId'] as int? ?? 0;
    final bankName = extra?['bankName'] as String? ?? '';  // 추가
    
    return TransferPage(
      accountNo: accountNo,
      withdrawalAccountId: withdrawalAccountId,
      bankName: bankName,  // 이 부분이 추가되어야 함
    );
  },
),
      // GoRoute(
      //   path: FinanceRoutes.transfer,
      //   builder: (context, state) => const FinanceTransferPage(), //////// 송금 테스트 페이지 였던것
      // ),
      GoRoute(
        path: FinanceRoutes.analysis,
        builder: (context, state) => const FinanceAnalysisPage(),
      ),
      GoRoute(
        path: FinanceRoutes.demandDetail,
        builder: (context, state) {
          final accountNo = state.pathParameters['accountNo'] ?? '';
          final extra = state.extra as Map<String, dynamic>?;
          
          return DemandDetailPage(
            accountNo: accountNo,
            bankName: extra?['bankName'] ?? '',
            accountName: extra?['accountName'] ?? '',
            balance: extra?['balance'] ?? 0,
            noMoneyMan: extra?['noMoneyMan'] ?? false,
          );
        },
      ),
      GoRoute(
        path: FinanceRoutes.depositDetail,
        builder: (context, state) {
          final accountNo = state.pathParameters['accountNo'] ?? '';
          final extra = state.extra as Map<String, dynamic>?;
          
          return DepositDetailPage(
            accountNo: accountNo,
            bankName: extra?['bankName'] ?? '',
            accountName: extra?['accountName'] ?? '',
            balance: extra?['balance'] ?? 0,
            noMoneyMan: extra?['noMoneyMan'] ?? false,
          );
        },
      ),
      GoRoute(
        path: FinanceRoutes.savingDetail,
        builder: (context, state) {
          final accountNo = state.pathParameters['accountNo'] ?? '';
          final extra = state.extra as Map<String, dynamic>?;
          
          return SavingDetailPage(
            accountNo: accountNo,
            bankName: extra?['bankName'] ?? '',
            accountName: extra?['accountName'] ?? '',
            balance: extra?['balance'] ?? 0,
            noMoneyMan: extra?['noMoneyMan'] ?? false,
          );
        },
      ),
      GoRoute(
        path: FinanceRoutes.loanDetail,
        builder: (context, state) {
          final accountNo = state.pathParameters['accountNo'] ?? '';
          final extra = state.extra as Map<String, dynamic>?;
          print("라우트까지옴");
          return LoanDetailPage(
            accountNo: accountNo,
            bankName: extra?['bankName'] ?? '',
            accountName: extra?['accountName'] ?? '',
            balance: extra?['balance'] ?? 0,
          );
        },
      ),
      GoRoute(
        path: FinanceRoutes.cardDetail,
        builder: (context, state) {
          final cardNo = state.pathParameters['cardNo'] ?? '';
          final extra = state.extra as Map<String, dynamic>?;
          
          return CardDetailPage(
            cardNo: cardNo,
            bankName: extra?['bankName'] ?? '',
            cardName: extra?['cardName'] ?? '',  // accountName 대신 cardName 사용
            cvc: extra?['cvc'] ?? '',  // cvc 추가
            balance: extra?['balance'] != null
              ? (extra?['balance'] is String
                  ? int.tryParse(extra?['balance']) ?? 0
                  : extra?['balance'])
              : 0,
          );
        },
      ),
      GoRoute(
        path: FinanceRoutes.withdrawalAccountRegistration,
        builder: (context, state) {
          final accountNo = state.pathParameters['accountNo'] ?? '';
          return WithdrawalAccountRegistrationPage(accountNo: accountNo);
        },
      ),
      // 인증 페이지 라우트 수정
      GoRoute(
        path: FinanceRoutes.auth,
        builder: (context, state) {
          final extra = state.extra as Map<String, dynamic>?;
          final accountNo = extra?['accountNo'] as String? ?? '';
          final withdrawalAccountId = extra?['withdrawalAccountId'] as int? ?? 0;
          
          return CertificateAuthPage(
            accountNo: accountNo,
            withdrawalAccountId: withdrawalAccountId,
          );
        },
      ),
      GoRoute(
        path: FinanceRoutes.agreement,
        builder: (context, state) {
          final agreementNo = state.pathParameters['agreementNo'] ?? '';
          return FinanceAgreementDetailPage(
            agreementNo : agreementNo,
          );
        }
      ),
      GoRoute(
        path: FinanceRoutes.transferComplete,
        builder: (context, state) {
          final extra = state.extra as Map<String, dynamic>?;
          final withdrawalAccountNo = extra?['withdrawalAccountNo'] as String? ?? '';
          final depositAccountNo = extra?['depositAccountNo'] as String? ?? '';
          final amount = extra?['amount'] as int? ?? 0;
          
          
          return TransferCompletePage(
            withdrawalAccountNo: withdrawalAccountNo,
            depositAccountNo: depositAccountNo,
            amount: amount,
          );
        },
      ),
    ],
  ),
];