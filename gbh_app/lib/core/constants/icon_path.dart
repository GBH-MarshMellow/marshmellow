import 'dart:io';

import 'package:flutter/material.dart';

class IconPath {
  // 앱바
  static const String analysis = 'assets/icons/app_bar/analysis.svg';
  static const String bell = 'assets/icons/app_bar/Bell.svg';
  static const String userCircle = 'assets/icons/app_bar/UserCircle.svg';
  static const String simpleUnclicked =
      'assets/icons/app_bar/simple_unclicked.svg';
  static const String simpleClicked = 'assets/icons/app_bar/simple_clicked.svg';

  // bank
  static const String koreaBank =
      'assets/icons/bank/001_korea.svg'; // 한국은행 (Korea Bank)
  static const String kdbBank =
      'assets/icons/bank/002_kdb.svg'; // 산업은행 (KDB Bank)
  static const String ibkBank =
      'assets/icons/bank/003_ibk.svg'; // 기업은행 (IBK Bank)
  static const String kbBank = 'assets/icons/bank/004_kb.svg'; // 국민은행 (KB Bank)
  static const String nhBank = 'assets/icons/bank/011_nh.svg'; // 농협은행 (NH Bank)
  static const String wooriBank =
      'assets/icons/bank/020_woori.svg'; // 우리은행 (Woori Bank)
  static const String scBank =
      'assets/icons/bank/023_sc.svg'; // SC은행 (Standard Chartered Bank)
  static const String citiBank =
      'assets/icons/bank/027_citi.svg'; // 씨티은행 (Citi Bank)
  static const String dgBank =
      'assets/icons/bank/032_daegu.svg'; // 대구은행 (Daegu Bank)
  static const String gjBank =
      'assets/icons/bank/034_gwangju.svg'; // 광주은행 (Gwangju Bank)
  static const String jejuBank =
      'assets/icons/bank/035_jeju.svg'; // 제주은행 (Jeju Bank)
  static const String jbBank =
      'assets/icons/bank/037_junbuk.svg'; // 전북은행 (Jeonbuk Bank)
  static const String gnBank =
      'assets/icons/bank/039_gyeongnam.svg'; // 경남은행 (Gyeongnam Bank)
  static const String mgBank =
      'assets/icons/bank/045_mg.svg'; // MG새마을금고 (MG Community Credit Cooperatives)
  static const String hanaBank =
      'assets/icons/bank/081_hana.svg'; // 하나은행 (Hana Bank)
  static const String shinhanBank =
      'assets/icons/bank/088_shinhan.svg'; // 신한은행 (Shinhan Bank)
  static const String kakaoBank =
      'assets/icons/bank/090_kakao.svg'; // 카카오뱅크 (Kakao Bank)
  static const String ssafyBank =
      'assets/icons/bank/999_ssafy.svg'; // 싸피뱅크 (SSAFY Bank)

  static const String koreaBank2 =
      'assets/icons/bank/001_korea_2.png'; // 한국은행 (Korea Bank)
  static const String ssafyBank2 =
      'assets/icons/bank/999_ssafy_2.png'; // 싸피뱅크 (SSAFY Bank)

  // expense
  static const String expenseAlcohol =
      'assets/icons/expense/expense_alcohol.svg';
  static const String expenseBaby = 'assets/icons/expense/expense_baby.svg';
  static const String expenseBank = 'assets/icons/expense/expense_bank.svg';
  static const String expenseCar = 'assets/icons/expense/expense_car.svg';
  static const String expenseCoffee = 'assets/icons/expense/expense_coffee.svg';
  static const String expenseCulture =
      'assets/icons/expense/expense_culture.svg';
  static const String expenseEvent = 'assets/icons/expense/expense_event.svg';
  static const String expenseFood = 'assets/icons/expense/expense_food.svg';
  static const String expenseHealth = 'assets/icons/expense/expense_health.svg';
  static const String expenseHouse = 'assets/icons/expense/expense_house.svg';
  static const String expenseLiving = 'assets/icons/expense/expense_living.svg';
  static const String expenseOnlineShopping =
      'assets/icons/expense/expense_onlineshopping.svg';
  static const String expensePet = 'assets/icons/expense/expense_pet.svg';
  static const String expenseShopping =
      'assets/icons/expense/expense_shopping.svg';
  static const String expenseStudy = 'assets/icons/expense/expense_study.svg';
  static const String expenseTransport =
      'assets/icons/expense/expense_transport.svg';
  static const String expenseTravel = 'assets/icons/expense/expense_travel.svg';
  static const String expenseBeauty = 'assets/icons/expense/expense_beauty.svg';
  static const String nonCategory = 'assets/icons/expense/non_category.svg';

  // income
  // 금융수입
  static const String incomeBank = 'assets/icons/income/income_bank.svg';
  //사업수입
  static const String incomeBusiness =
      'assets/icons/income/income_business.svg';
  //보험금
  static const String incomeInsurance =
      'assets/icons/income/income_insurance.svg';
  //더치페이
  static const String incomeNpay = 'assets/icons/income/income_npay.svg';
  //아르바이트
  static const String incomeParttime =
      'assets/icons/income/income_parttime.svg';
  //부동산
  static const String incomeRealestate =
      'assets/icons/income/income_realestate.svg';
  //중고거래
  static const String incomeRecycle = 'assets/icons/income/income_recycle.svg';
  //월급
  static const String incomeSalary = 'assets/icons/income/income_salary.svg';
  //장학금
  static const String incomeScholarship =
      'assets/icons/income/income_scholarship.svg';
  //SNS
  static const String incomeSns = 'assets/icons/income/income_sns.svg';
  //기타
  static const String incomeEtc = 'assets/icons/income/non_category.svg';

  // 이체
  //기타
  static const String transferEtc = 'assets/icons/transfer/non_category.svg';
  // 입금
  static const String paid = 'assets/icons/transfer/paid.svg';
  // 출금
  static const String received = 'assets/icons/transfer/received.svg';

  // nav
  static const String budgetBk = 'assets/icons/nav/budget_bk.svg';
  static const String budgetWh = 'assets/icons/nav/budget_wh.svg';
  static const String cookieBk = 'assets/icons/nav/cookie_bk.svg';
  static const String cookieWh = 'assets/icons/nav/cookie_wh.svg';
  static const String financeBk = 'assets/icons/nav/finance_bk.svg';
  static const String financeWh = 'assets/icons/nav/finance_wh.svg';
  static const String ledgerBk = 'assets/icons/nav/ledger_bk.svg';
  static const String ledgerWh = 'assets/icons/nav/ledger_wh.svg';
  static const String userBk = 'assets/icons/nav/user_bk.svg';
  static const String userWh = 'assets/icons/nav/user_wh.svg';

  // search_bar
  static const String searchButton =
      'assets/icons/search_bar/search_button.svg';
  static const String searchButtonPng =
      'assets/icons/search_bar/search_button.png';
  static const String exitgray = 'assets/icons/search_bar/exitgray.svg';

  //body
  static const String caretLeft = 'assets/icons/body/CaretLeft.svg';
  static const String caretRight = 'assets/icons/body/CaretRight.svg';
  static const String plus = 'assets/icons/body/plus.svg';
  static const String searchOutlined = 'assets/icons/body/search_outlined.svg';
  static const String caretDown = 'assets/icons/body/CaretDown.svg';
  static const String pencilSimple = 'assets/icons/body/PencilSimple.svg';
  static const String quoteLeft = 'assets/icons/body/Quotes_L.svg';
  static const String quoteRight = 'assets/icons/body/Quotes_R.svg';
  static const String refesh = 'assets/icons/body/ArrowCounterClockwise.svg';
  static const String map = 'assets/icons/body/Map.svg';
  static const String tent = 'assets/icons/body/Tent.svg';
  static const String rocket = 'assets/icons/body/Rocket.svg';
  static const String gas = 'assets/icons/body/Gas.svg';

  static const String caretcircleup = 'assets/icons/body/CaretCircleUp.svg';
  static const String caretdoubledown = 'assets/icons/body/CaretDoubleDown.svg';
  static const String question = 'assets/icons/body/question.svg';
  static const String add = 'assets/icons/body/Add.svg';
  static const String shareNetwork = 'assets/icons/body/ShareNetwork.svg';
  static const String paperclip = 'assets/icons/body/Paperclip.svg';

  static const String filtered = 'assets/icons/body/filtered.svg';
  static const String unfiltered = 'assets/icons/body/unfiltered.svg';

  // 카드 이미지
  static const String testCard = 'assets/icons/card/testCard.svg';

  // 축하 컨페티
  static const String confettiGreen =
      'assets/images/celebration/confetti_green.png';
  static const String confettiPink =
      'assets/images/celebration/confetti_pink.png';
  static const String confettiBlue =
      'assets/images/celebration/confetti_blue.png';
  static const String confettiYellow1 =
      'assets/images/celebration/confetti_yellow1.png';
  static const String confettiYellow2 =
      'assets/images/celebration/confetti_yellow2.png';

  // 파일
  static const String fileCsv = 'assets/icons/files/FileCsv.svg';
  static const String fileDoc = 'assets/icons/files/FileDoc.svg';
  static const String fileJpg = 'assets/icons/files/FileJpg.svg';
  static const String filePdf = 'assets/icons/files/FilePdf.svg';
  static const String filePng = 'assets/icons/files/FilePng.svg';
  static const String filePpt = 'assets/icons/files/FilePpt.svg';
  static const String fileSvg = 'assets/icons/files/FileSvg.svg';
  static const String fileText = 'assets/icons/files/FileText.svg';
  static const String fileTxt = 'assets/icons/files/FileTxt.svg';
  static const String fileXls = 'assets/icons/files/FileXls.svg';
  static const String fileZip = 'assets/icons/files/FileZip.svg';

  // 폴더
  static const String folderSimple = 'assets/icons/files/folderSimple.svg';

  // 포트폴리오 체크
  static const String uncheckedFolder =
      'assets/icons/files/unchecked_folder.svg';
  static const String checkedFolder = 'assets/icons/files/checked_folder.svg';
  static const String checked = 'assets/icons/files/checked.svg';
  static const String unchecked = 'assets/icons/files/unchecked.svg';
  static const String trash = 'assets/icons/files/trash.svg';

  // 퇴사 망상
  static const String card1 = 'assets/images/quit/card1.png';
  static const String card2 = 'assets/images/quit/card2.png';
  static const String card3 = 'assets/images/quit/card3.png';
  static const String card4 = 'assets/images/quit/card4.png';
  static const String card5 = 'assets/images/quit/card5.png';
  static const String card6 = 'assets/images/quit/card6.png';
  static const String card7 = 'assets/images/quit/card7.png';
}
