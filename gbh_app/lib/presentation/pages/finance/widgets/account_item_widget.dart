// presentation/pages/finance/widgets/account_item_widget.dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/finance/services/transfer_service.dart';
import 'package:marshmellow/presentation/widgets/finance/bank_icon.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';


class AccountItemWidget extends ConsumerWidget {
  final String bankName; // 은행명
  final String accountName; // 계좌 명
  final String accountNo; // 계좌번호
  final int balance; // 계좌잔액
  final bool isLoan; // 대출정보 여부
  final bool noMoneyMan; // 송금가능여부
  final String type; // 입출금, 예금, 적금, 대출

  const AccountItemWidget({
    Key? key,
    required this.bankName,
    required this.accountName,
    required this.accountNo,
    required this.balance,
    this.isLoan = false,
    this.noMoneyMan = false,
    required this.type,
  }) : super(key: key);

  // 숫자 포맷팅 함수 (천 단위 구분)
  String formatAmount(int amount) {
    final formatter = NumberFormat('#,###');
    return formatter.format(amount);
  }

  // AccountItemWidget.dart의 onTap 처리 메서드 예시
  void _onAccountItemTap(BuildContext context) {
    print("계좌 클릭: 유형=$type, 계좌번호=$accountNo");
    // type에 따라 다른 경로로 이동
    switch (type) {
      case '입출금':
        context.push(
          FinanceRoutes.getDemandDetailPath(accountNo),
          extra: {
            'bankName': bankName,
            'accountName': accountName,
            'accountNo': accountNo,
            'balance': balance,
            'noMoneyMan': noMoneyMan,
          },
        );
        break;
      case '예금':
        // 예금 상세 페이지로 이동 (아직 구현되지 않음)
        context.push(
          FinanceRoutes.getDepositDetailPath(accountNo),
          extra: {
            'bankName': bankName,
            'accountName': accountName,
            'accountNo': accountNo,
            'balance': balance,
            'noMoneyMan': noMoneyMan,
          },
        );
        break;
      case '적금':
        // 적금 상세 페이지로 이동 (아직 구현되지 않음)
        context.push(
          FinanceRoutes.getSavingDetailPath(accountNo),
          extra: {
            'bankName': bankName,
            'accountName': accountName,
            'accountNo': accountNo,
            'balance': balance,
            'noMoneyMan': noMoneyMan,
          },
        );
        break;
      case '대출':
        // 대출 상세 페이지로 이동 (아직 구현되지 않음)
        print("대출 path이동시도 ");
        context.push(
          FinanceRoutes.getLoanDetailPath(accountNo),
          extra: {
            'bankName': bankName,
            'accountName': accountName,
            'accountNo': accountNo,
            'balance': balance,
            'noMoneyMan': noMoneyMan,
          },
        );
        break;
      default:
        // 기본 처리 (필요시)
        break;
    }
  }




  @override
  Widget build(BuildContext context, WidgetRef ref) {
  final isHidden = ref.watch(isFinanceHideProvider);
  return Container(
    margin: const EdgeInsets.only(bottom: 8),
    decoration: BoxDecoration(
      border: Border.all(color: Colors.grey.shade300),
      borderRadius: BorderRadius.circular(8),
    ),
    child: ClipRRect( // 효과를 컨테이너 내부로 제한
      borderRadius: BorderRadius.circular(8),
      child: Material( // Material 위젯 추가
        color: Colors.transparent,
        child: InkWell(
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 상세페이지 테스트중 <<<<<<<<<<<<<<<<<<<<<<<<<
          onTap: () {
            _onAccountItemTap(context);
          },
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 상세페이지 테스트중 >>>>>>>>>>>>>>>>>>>>>>>>>>
        child: Padding(
          padding: const EdgeInsets.fromLTRB(8.0, 12.0, 14.0, 12.0),
          child: Row(
            children: [
              Padding(
                padding: const EdgeInsets.all(8),
                child: BankIcon(bankName: bankName, size: 40),
              ),
              // Text(bankName,style: AppTextStyles.bodySmall.copyWith(fontWeight: FontWeight.w600)),

              const SizedBox(width: 2),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Text('타입: $type'),
                    Text(
                      accountName,
                      style: AppTextStyles.bodySmall.copyWith(fontWeight: FontWeight.w400,
                      color: AppColors.blackLight)
                    ),
                    // if (bankName != '-') Text('은행: $bankName'),
                    // Text('계좌번호: ${_maskAccountNumber(accountNo)}'),
                    Text(
                      isLoan 
                        ? isHidden ? '대출금액 보기' : '대출금액 ${formatAmount(balance)}원' 
                        : isHidden ? '잔액보기' : '${formatAmount(balance)}원',
                      style: isHidden ? AppTextStyles.bodyMediumLight : AppTextStyles.subTitle
                    ),
                  ],
                ),
              ),
              noMoneyMan // 입출금 계좌 여부 판단 변수 
                ? GestureDetector(
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 상세페이지 테스트중 <<<<<<<<<<<<<<<<<<<<<<<<<
                    onTap: () {
                      // 송금 불가능한 상태라면 처리하지 않음
                      if (!noMoneyMan) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('송금 가능한 계좌가 아닙니다.')),
                        );
                        return;
                      }
                      // 송금 버튼 핸들러 호출
                      TransferService.handleTransfer(context, ref, accountNo, bankName);
                    },
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 상세페이지 테스트중 >>>>>>>>>>>>>>>>>>>>>>>>>>
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
                      // margin: EdgeInsets.fromLTRB(0.0, 0.0, 6.0, 0.0),
                      decoration: BoxDecoration(
                        color: AppColors.buttonBlack,
                        borderRadius: BorderRadius.circular(6.0),
                      ),
                      child: Text(
                        "송금",
                        style: AppTextStyles.bodySmall.copyWith(color: AppColors.background)
                        // style: AppTextStyles.buttonBold.copyWith(color: AppColors.background)
                        // style: AppTextStyles.button.copyWith(color: AppColors.background)
                      ),
                    ),
                  )
                : IconButton(
                    icon: Icon(Icons.arrow_forward_ios, size: 16),
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 상세페이지 테스트중 <<<<<<<<<<<<<<<<<<<<<<<<<
                    onPressed: () {
                      _onAccountItemTap(context);
                    },
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 상세페이지 테스트중 >>>>>>>>>>>>>>>>>>>>>>>>>>
                  ),
            ],
          ),),),
        ),
      ),
    );
  }

}