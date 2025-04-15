// lib/router/routes/signup_routes.dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/data/models/wishlist/wishlist_model.dart';
import 'package:marshmellow/presentation/pages/auth/signup_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/mydata/auth_mydata_agreement_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/mydata/auth_mydata_already_connected_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/mydata/auth_mydata_cert_completion_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/mydata/auth_mydata_cert_email_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/mydata/auth_mydata_cert_login_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/mydata/auth_mydata_cert_pw_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/mydata/auth_mydata_splash_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/pinnum/auth_pinnum_create_complete_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/message/auth_message_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/pinnum/auth_pinnum_create_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/message/auth_message_verification_loading_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/message/auth_message_complete_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/pinnum/auth_pinnum_login_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/salary_input/salary_input_complete_page.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/salary_input/salary_input_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_salary/budget_type_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_salary/budget_type_selection_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_salary/salary_celebrate_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/salary_to_wish/wish_complete_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/salary_to_wish/wish_selection_page.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/salary_to_wish/wish_setup_page.dart';

/*
  회원가입 routes
*/
class SignupRoutes {
  static const String root = '/signup';
  // 추가 경로가 필요하면 여기에 정의
  static const String authmessage = 'authmessage';
  static const String authloading = 'authloading';
  static const String authcomplete = 'authcomplete';
  static const String pinsetup = 'pinsetup';
  static const String pincomplete = 'pincomplete';
  static const String pinlogin = 'pinlogin';
  static const String pinlogincomplete = 'pinlogincomplete';
  static const String mydatasetup = 'mydatasetup';
  static const String mydataemail = 'mydataaemail';
  static const String mydatapassword = 'mydatapassword';
  static const String mydatacomplete = 'mydatacomplete';
  static const String mydataselect = 'mydataselect';
  static const String mydatalogin = 'mydatalogin';
  static const String mydataagreement = 'mydataagreement';
  static const String mydataalreadyconn = 'mydataalreadyconn';
  static const String salaryinput = 'salaryinput';
  static const String salaryinputcomplete = 'salaryinputcomplete';

  // 위시 생성 경로 정의
  static const String wishcreate = 'wish/create';
  static const String wishsetup = 'wish/setup';
  static const String wishcomplete = 'wish/complete';

  // 예산 분배 유형 선택 경로 정의
  static const String budgettype = 'type';
  static const String budgettypeselection = 'type/selection';
  static const String budgetcelebrate = 'celebrate';

  // 전체 경로 생성 헬퍼 메서드
  static String getAuthMessagePath() => '$root/$authmessage';
  static String getAuthLoadingPath() => '$root/$authloading';
  static String getAuthCompletePath() => '$root/$authcomplete';
  static String getPinSetupPath() => '$root/$pinsetup';
  static String getPinCompletePath() => '$root/$pincomplete';
  static String getPinLoginPath() => '$root/$pinlogin';
  static String getPinLoginCompletePath() => '$root/$pinlogincomplete';
  static String getMyDataSplashPath() => '$root/$mydatasetup';
  static String getMyDataEmailPath() => '$root/$mydataemail';
  static String getMyDataPasswordPath() => '$root/$mydatapassword';
  static String getMyDataCompletePath() => '$root/$mydatacomplete';
  static String getMyDataSelectPath() => '$root/$mydataselect';
  static String getMyDataLoginPath() => '$root/$mydatalogin';
  static String getMyDataAgreementPath() => '$root/$mydataagreement';
  static String getMyDataAlreadyConnPath() => '$root/$mydataalreadyconn';
  static String getSalaryInputPath() => '$root/$salaryinput';
  static String getSalaryInputCompletePath() => '$root/$salaryinputcomplete';

  // 위시 생성 경로 생성 헬퍼 메서드
  static String getWishCreatePath() => '$root/$wishcreate';
  static String getWishSetUpPath() => '$root/$wishsetup';
  static String getWishCompletePath() => '$root/$wishcomplete';

  // 예산 분배 경로 생성 헬퍼 메서드
  static String getBudgetTypePath() => '$root/$budgettype';
  static String getBudgetTypeSelectionPath() => '$root/$budgettypeselection';
  static String getBudgetCelebratePath() => '$root/$budgetcelebrate';
}

List<RouteBase> signupRoutes = [
  GoRoute(
    path: SignupRoutes.root,
    builder: (context, state) => const SignupPage(),
    routes: [
      // 인증 메시지 페이지
      GoRoute(
        path: SignupRoutes.authmessage,
        builder: (context, state) {
          final Map<String, dynamic> extra = state.extra as Map<String, dynamic>;
          return AuthMessagePage(userInfo:  extra);
        },
      ),

      // 본인인증 진행중 페이지
      GoRoute(
        path: SignupRoutes.authloading,
        builder: (context, state) => const AuthVerificationLoadingPage(),
      ),
      
      // 본인인증 완료 페이지
      GoRoute(
        path: SignupRoutes.authcomplete,
        builder: (context, state) => const AuthCompletePage(),
      ),

      // 핀번호 생성 페이지
      GoRoute(
        path: SignupRoutes.pinsetup,
        builder: (context, state) => const AuthPinnumPage(),
      ),

      // 핀번호 완료 페이지
      GoRoute(
        path: SignupRoutes.pincomplete,
        builder: (context, state) => const AuthPinnumCompletePage(),
      ),

      // 핀번호 로그인 페이지
      GoRoute(
        path: SignupRoutes.pinlogin,
        builder: (context, state) => const AuthPinnumLoginPage(),
      ),

      // 마이데이터 신규 생성 페이지
      GoRoute(
        path: SignupRoutes.mydatasetup,
        builder: (context, state) => const AuthMydataSplashPage(),
      ),

      // 마이데이터 이메일 작성 페이지
      GoRoute(
        path: SignupRoutes.mydataemail,
        builder: (context, state) => const AuthMydataEmailInputPage(),
      ),

      // 마이데이터 비밀번호 설정 페이지
      GoRoute(
        path: SignupRoutes.mydatapassword,
        builder: (context, state) => const AuthMydataCertPwPage(),
      ),

      // 마이데이터 인증서 생성 완료 페이지
      GoRoute(
        path: SignupRoutes.mydatacomplete,
        builder: (context, state) => const AuthMydataCertCompletePage(),
      ),

      // 마이데이터 로그인 페이지
      GoRoute(
        path: SignupRoutes.mydatalogin,
        builder: (context, state) => const AuthMydataCertLoginPage(),
      ),

      // 이미 연동된 사용자 페이지
      GoRoute(
        path: SignupRoutes.mydataalreadyconn,
        builder: (context, state) => const AuthAlreadyConnectedPage(),
      ),

      // 전자서명 원문 페이지
      GoRoute(
        path: SignupRoutes.mydataagreement,
        builder: (context, state) => const AuthMydataAgreementPage(),
      ),
      
      // 월급 정보 입력 페이지
      GoRoute(
        path: SignupRoutes.salaryinput,
        builder: (context, state) => const SalaryInputPage(),
      ),

      // 월급 정보 입력 완료 확인페이지
      GoRoute(
        path: SignupRoutes.salaryinputcomplete,
        builder: (context, state) => const SalaryInputCompletePage(),
      ),

      // 예산 월급날 축하 페이지
      GoRoute(
        path: SignupRoutes.budgetcelebrate,
        builder: (context, state) {
          return SalaryCelebratePage();
        },
      ),


      // 예산 분배 유형 출력 페이지
      GoRoute(
        path: SignupRoutes.budgettype,
        builder: (context, state) {
          return BudgetTypePage();
        },
      ),

      // 예산 분배 유형 선택 페이지
      GoRoute(
        path: SignupRoutes.budgettypeselection,
        builder: (context, state) {
          return BudgetTypeSelectionPage();
        },
      ),

            // 위시 생성 페이지
      GoRoute(
        path: SignupRoutes.wishcreate,
        builder: (context, state) {
          return WishSelectionPage();
        },
      ),

      // 위시 날짜 및 계좌 설정 페이지
      GoRoute(
        path: SignupRoutes.wishsetup,
        builder: (context, state) {
          final wishlist = state.extra as Wishlist;
          return WishSetupPage(wishlist: wishlist);
        },
      ),

      // 위시 생성 완료 페이지
      GoRoute(
        path: SignupRoutes.wishcomplete,
        builder: (context, state) {
          final extra = state.extra as Map<String, dynamic>;
          return WishCompletePage(      
            wishlist: extra['wishlist'],
            selectedMonth: extra['selectedMonth'],
            dailyAmount: extra['dailyAmount'],
            withdrawalAccount: extra['withdrawalAccount'],
            depositAccount: extra['depositAccount'],
            dueDate: extra['dueDate'],
          );
        },
      ),



    ],
  ),
];
