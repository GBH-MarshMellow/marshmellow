import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:go_router/go_router.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/utils/format_utils.dart';
import 'package:marshmellow/data/models/finance/transfer_model.dart';
import 'package:marshmellow/presentation/viewmodels/finance/demand_detail_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/finance/transfer_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/keyboard/index.dart';
import 'package:marshmellow/presentation/widgets/loading/loading_manager.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';
import 'package:flutter/services.dart';

class TransferPage extends ConsumerStatefulWidget {
  final String accountNo;
  final int withdrawalAccountId; // 추가된 필드
  final String bankName;  // 추가
  
  const TransferPage({
    Key? key,
    required this.accountNo,
    required this.withdrawalAccountId, // 생성자 매개변수 추가
    required this.bankName,  // 추가
  }) : super(key: key);

  @override
  ConsumerState<TransferPage> createState() => _TransferPageState();
}

class _TransferPageState extends ConsumerState<TransferPage> {
  final TextEditingController _accountController = TextEditingController();
  final TextEditingController _amountController = TextEditingController();
  final TextEditingController _bankController = TextEditingController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(transferProvider.notifier).reset();
      ref.read(transferProvider.notifier).setWithdrawalAccount(widget.withdrawalAccountId, widget.accountNo);
    });
  }

  @override
  void dispose() {
    _accountController.dispose();
    _amountController.dispose();
    _bankController.dispose();
    super.dispose();
  }

  // 은행 선택 모달
  void _showBankSelectionModal(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (BuildContext context) {
        return Container(
          padding: const EdgeInsets.all(20),
          height: MediaQuery.of(context).size.height * 0.7,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                '은행 선택',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 20),
              Expanded(
                child: ListView.builder(
                  itemCount: bankList.length,
                  itemBuilder: (context, index) {
                    final bank = bankList[index];
                    return ListTile(
                      leading: bank.code == '001' || bank.code == '999' ? 
                        Image.asset(
                          bank.code == '001' ? 'assets/icons/bank/001_korea_2.png' : 'assets/icons/bank/999_ssafy_2.png',
                          width: 30,
                          height: 30,
                        ) : 
                        SvgPicture.asset(
                          bank.iconPath,
                          width: 30,
                          height: 30,
                        ),
                      title: Text(bank.name),
                      onTap: () {
                        ref.read(transferProvider.notifier).selectBank(
                          bank.code,
                          bank.name,
                        );
                        _bankController.text = bank.name;
                        Navigator.pop(context);
                      },
                    );
                  }
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(transferProvider);
    final viewModel = ref.read(transferProvider.notifier);

    // 에러 메시지 표시
    if (state.error != null) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(state.error!)),
        );
      });
    }

    return WillPopScope(
      onWillPop: () async {
        viewModel.reset();
        return true;
      },
      child: Scaffold(
        appBar: CustomAppbar(
          title: 'my little 자산',
          backgroundColor: AppColors.background,
        ),
        body: _buildBody(context, state, viewModel),
      ),
    );
  }

  Widget _buildBody(BuildContext context, TransferState state, TransferViewModel viewModel) {
    switch (state.step) {
      case TransferStep.accountInput:
        return _buildAccountInputStep(context, state, viewModel);
      case TransferStep.amountInput:
        return _buildAmountInputStep(context, state, viewModel);
      case TransferStep.loading:
        return const Center(child: CircularProgressIndicator());
      case TransferStep.complete:
        return _buildCompleteStep(context, state, viewModel);
      default:
        return const Center(child: CircularProgressIndicator());
    }
  }

  // 계좌 정보 입력 단계
// 계좌 정보 입력 단계
  Widget _buildAccountInputStep(BuildContext context, TransferState state, TransferViewModel viewModel) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const SizedBox(height: 20),
          
          // 출금계좌 정보 (변경 불가)
          Container(
            padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
            decoration: BoxDecoration(
              color:AppColors.whiteDark,
              borderRadius: BorderRadius.circular(5),
            ),
            child:                 
                Text(
                  '출금계좌 : ${widget.bankName} ${widget.accountNo}',
                  style: AppTextStyles.bodyMedium.copyWith(color: AppColors.divider)
                ),
          ),
          const SizedBox(height: 20),
          
          // 은행 선택
          Container(
            decoration: BoxDecoration(
              border: Border.all(color: Colors.grey.shade300),
              borderRadius: BorderRadius.circular(5),
            ),
            child: TextField(
              controller: _bankController,
              style: AppTextStyles.bodyMedium,
              readOnly: true,
              decoration: InputDecoration(
                labelText: '은행/기관',
                floatingLabelBehavior: FloatingLabelBehavior.always,
                labelStyle: TextStyle(color: Colors.grey.shade600, fontSize: 14),
                contentPadding: const EdgeInsets.symmetric(horizontal: 15, vertical: 15),
                border: InputBorder.none,
                suffixIcon: const Icon(Icons.keyboard_arrow_down , color: AppColors.blackPrimary,),
              ),
              onTap: () => _showBankSelectionModal(context),
            ),
          ),
          const SizedBox(height: 10),
          
          // 계좌번호 입력
          Container(
            decoration: BoxDecoration(
              border: Border.all(color: Colors.grey.shade300),
              borderRadius: BorderRadius.circular(5),
            ),
            child: TextField(
              controller: _accountController,
              style: AppTextStyles.bodyMedium,
              decoration: InputDecoration(
                labelText: '입금 계좌번호',
                floatingLabelBehavior: FloatingLabelBehavior.always,
                labelStyle: TextStyle(color: Colors.grey.shade600, fontSize: 14),
                contentPadding: const EdgeInsets.symmetric(horizontal: 15, vertical: 15),
                border: InputBorder.none,
              ),
              keyboardType: TextInputType.number,
              inputFormatters: [
                AccountNumberFormatter(),
              ],
              onChanged: (value) {
                // 하이픈을 제거한 실제 계좌번호를 뷰모델에 전달
                viewModel.setDepositAccountNo(value.replaceAll('-', ''));
              },
            ),
          ),
          
          const Spacer(),
          
          // 버튼
          Container(
            margin: const EdgeInsets.only(bottom: 20),
            child: 
              SizedBox(
                width: MediaQuery.of(context).size.width,
                child: ElevatedButton(
                  onPressed: state.isAccountInputComplete 
                    ? viewModel.moveToAmountInput 
                    : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.buttonBlack,
                    padding: const EdgeInsets.symmetric(vertical: 15),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(5),
                    ),
                  ),
                  child: const Text(
                    '다음', 
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                    ),
                  ),
                ),
            ),
          ),
        ],
      ),
    );
  }
  // 금액 입력 단계
  Widget _buildAmountInputStep(BuildContext context, TransferState state, TransferViewModel viewModel) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const SizedBox(height: 20),
          
          // 출금계좌 정보 (변경 불가)
          Container(
            padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
            decoration: BoxDecoration(
              color:AppColors.whiteDark,
              borderRadius: BorderRadius.circular(5),
            ),
            child:                 
              Text(
                '출금계좌 : ${widget.bankName} ${widget.accountNo}',
                style: AppTextStyles.bodyMedium.copyWith(color: AppColors.divider)
              ),
          ),
          const SizedBox(height: 20),
          
          // 보낼 금액 표시 또는 입력된 금액 표시
          GestureDetector(
            onTap: () async {
              await KeyboardModal.showNumericKeyboard(
                context: context,
                onValueChanged: (value) {
                  setState(() {
                    _amountController.text = value;
                    // 예외 처리 추가
                    if (value.isNotEmpty && value != ',') {
                      print('설정할 금액: $value');
                      viewModel.setAmount(int.tryParse(value.replaceAll(',', '')) ?? 0);
                    } else {
                      viewModel.setAmount(0);
                    }
                  });
                },
                initialValue: _amountController.text.isEmpty ? '' : _amountController.text,
              );
            },
            child: Text(
              _amountController.text.isEmpty ? '보낼 금액' : NumberFormat.formatWithComma(_amountController.text.replaceAll(',', '')) + '원',
              style: AppTextStyles.bodyExtraLarge,
            ),
          ),
          const SizedBox(height: 20),

          // 입금계좌 정보 (변경 불가)
          Container(
            padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 12),
            decoration: BoxDecoration(
              color:AppColors.whiteDark,
              borderRadius: BorderRadius.circular(5),
            ),
            child:                 
              Text(
                '입금계좌 : ${state.selectedBankName} ${state.depositAccountNo}',
                style: AppTextStyles.bodyMedium.copyWith(color: AppColors.divider)
              ),
          ),
          
          const Spacer(),
          
          // 송금 버튼
          Button(
            onPressed: state.isAmountValid 
              ? () => context.push(
                  FinanceRoutes.getAuthPath(), 
                  extra: {
                    'accountNo': widget.accountNo, 
                    'withdrawalAccountId': widget.withdrawalAccountId,
                  }
                )
              : null,
            text: '송금하기',
            color: state.isAmountValid ? AppColors.blackPrimary : AppColors.whiteDark,
          ),
        ],
      ),
    );
  }

  // 송금 완료 단계
  Widget _buildCompleteStep(BuildContext context, TransferState state, TransferViewModel viewModel) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Lottie.asset(
            'assets/images/loading/success.json',
            width: 140,  // 원하는 크기로 조정
            height: 140,
            fit: BoxFit.contain,
          ),
          const SizedBox(height: 20),
          Text('${NumberFormat.formatWithComma(state.amount.toString())}원',style: AppTextStyles.bodyLarge.copyWith(fontWeight: FontWeight.w600)),
          const Text(
            '송금이 완료되었습니다',
            style: AppTextStyles.bodyLarge
          ),
          const SizedBox(height: 30),
          
          // 송금 정보 요약
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              // color: Colors.grey.shade100,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(width: 1.0 ,color: AppColors.blackLight)
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // const Text('송금 정보', style: TextStyle(fontWeight: FontWeight.bold)),
                // const Divider(),
                // const SizedBox(height: 10),
                
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text('출금 계좌', style: AppTextStyles.bodyMedium.copyWith(color: AppColors.disabled, fontWeight:  FontWeight.w400),),
                    Text(formatAccountNumber(state.withdrawalAccountNo) , style: AppTextStyles.bodyMedium),
                  ],
                ),
                const SizedBox(height: 8),
                
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text('입금 계좌', style: AppTextStyles.bodyMedium.copyWith(color: AppColors.disabled, fontWeight:  FontWeight.w400),),
                    Text(formatAccountNumber(state.depositAccountNo), style: AppTextStyles.bodyMedium),
                  ],
                ),
                const SizedBox(height: 8),
                
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text('보낸 금액', style: AppTextStyles.bodyMedium.copyWith(color: AppColors.disabled, fontWeight:  FontWeight.w400),),
                    Text('${NumberFormat.formatWithComma(state.amount.toString())}원', style: AppTextStyles.bodyMedium),
                  ],
                ),
              ],
            ),
          ),
          
          const Spacer(),
          
          // 확인 버튼
          Button(
            onPressed: () {
              // 페이지로 이동하기 전에 최신 계좌 정보를 가져옴
              final financeViewModel = ref.read(financeViewModelProvider.notifier);
              
              // 계좌 정보 새로고침 (이미 refreshAssetInfo가 호출되었지만 확실히 하기 위해)
              financeViewModel.refreshAssetInfo().then((_) {
                // Provider 캐시 무효화 다시 한번 확인
                ref.invalidate(demandTransactionsProvider);
                
                viewModel.reset();
                // 계좌 상세 페이지로 이동
                context.go(FinanceRoutes.root);
              });
            },
            text: '확인',
          ),
          SizedBox(height: 10,)
        ],
      ),
    );
  }
}

// 계좌번호 4자리씩 "-"
class AccountNumberFormatter extends TextInputFormatter {
  @override
  TextEditingValue formatEditUpdate(
    TextEditingValue oldValue,
    TextEditingValue newValue,
  ) {
    // 숫자만 추출
    String numbersOnly = newValue.text.replaceAll(RegExp(r'[^0-9]'), '');
    
    // 결과 문자열을 저장할 변수
    String formatted = '';
    
    // 4자리씩 하이픈 추가
    for (int i = 0; i < numbersOnly.length; i++) {
      // 4자리마다 하이픈 추가 (맨 앞은 제외)
      if (i > 0 && i % 4 == 0) {
        formatted += '-';
      }
      formatted += numbersOnly[i];
    }
    
    // 새로운 선택 위치 계산
    int newCursorPosition = newValue.selection.baseOffset + (formatted.length - newValue.text.length);
    // 유효하지 않은 커서 위치 보정
    if (newCursorPosition < 0) {
      newCursorPosition = 0;
    } else if (newCursorPosition > formatted.length) {
      newCursorPosition = formatted.length;
    }
    
    return TextEditingValue(
      text: formatted,
      selection: TextSelection.collapsed(offset: newCursorPosition),
    );
  }
}
String formatAccountNumber(String accountNo) {
  String numbersOnly = accountNo.replaceAll(RegExp(r'[^0-9]'), '');
  String formatted = '';
  
  for (int i = 0; i < numbersOnly.length; i++) {
    if (i > 0 && i % 4 == 0) {
      formatted += '-';
    }
    formatted += numbersOnly[i];
  }
  
  return formatted;
}