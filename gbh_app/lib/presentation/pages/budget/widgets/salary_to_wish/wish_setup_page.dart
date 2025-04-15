import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/wishlist/wish_model.dart';
import 'package:marshmellow/data/models/wishlist/wishlist_model.dart';
import 'package:marshmellow/presentation/viewmodels/budget/wish_selection_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/finance/bank_icon.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';
import 'package:marshmellow/router/routes/budget_routes.dart';
import 'package:lottie/lottie.dart';

// 이 파일 내에서 선택된 기간, 출금계좌, 입금계좌 상태를 관리하는 프로바이더들
final selectedMonthProvider = StateProvider<int>((ref) => 1);
final selectedWithdrawalAccountProvider = StateProvider<DemDepItem?>((ref) => null);
final selectedDepositAccountProvider = StateProvider<DemDepItem?>((ref) => null);

class WishSetupPage extends ConsumerStatefulWidget {
  final Wishlist wishlist;

  const WishSetupPage({Key? key, required this.wishlist}) : super(key: key);

  @override
  _WishSetupPageState createState() => _WishSetupPageState();
}

class _WishSetupPageState extends ConsumerState<WishSetupPage> {
  int _currentStep = 0;
  late List<int> _availableMonths;
  late List<int> _dailyAmounts;

  @override
  void initState() {
    super.initState();
    
    // 가능한 개월 수 계산 (1~6개월)
    _availableMonths = [1, 2, 3, 4, 5, 6];
    
    // 일별 금액 계산
    _calculateDailyAmounts();
    
    // 입출금 계좌 목록 로드
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(wishSelectionProvider.notifier).fetchDemDepList();
    });
  }

  void _calculateDailyAmounts() {
    // 일별 금액 계산 (각 개월 수로 나눈 금액)
    _dailyAmounts = _availableMonths.map((month) {
      // 30일 기준 일별 금액 계산
      return (widget.wishlist.productPrice / (month * 30)).round();
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final selectedMonth = ref.watch(selectedMonthProvider);
    final selectedWithdrawalAccount = ref.watch(selectedWithdrawalAccountProvider);
    final selectedDepositAccount = ref.watch(selectedDepositAccountProvider);
    final wishSelectionState = ref.watch(wishSelectionProvider);
    
    return Scaffold(
      appBar: CustomAppbar(
        title: '위시 설정',
        automaticallyImplyLeading: false,
      ),
      body: Stepper(
        currentStep: _currentStep,
        onStepContinue: () {
          if (_currentStep < 2) {
            // 현재 단계별 유효성 검사
            bool canContinue = false;
            
            switch (_currentStep) {
              case 0:
                // 개월 수 선택 단계
                canContinue = selectedMonth > 0;
                break;
              case 1:
                // 출금계좌 선택 단계
                canContinue = selectedWithdrawalAccount != null;
                break;
              case 2:
                // 입금계좌 선택 단계
                canContinue = selectedDepositAccount != null;
                break;
            }
            
            if (canContinue) {
              setState(() {
                _currentStep += 1;
              });
            }
          } else {
            // 마지막 단계에서는 API 호출 및 완료 페이지로 이동
            _registerWishAndAutoTransfer();
          }
        },
        onStepCancel: () {
          if (_currentStep > 0) {
            setState(() {
              _currentStep -= 1;
            });
          }
        },
        steps: [
          // 개월 수 선택 단계
          Step(
            title: Text('기간 선택', style: AppTextStyles.bodyLarge),
            content: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${widget.wishlist.productNickname} 구매를 위해 모을 기간을 선택해주세요',
                  style: AppTextStyles.bodyMedium,
                ),
                const SizedBox(height: 16),
                _buildMonthSelection(),
              ],
            ),
            isActive: _currentStep >= 0,
          ),
          
          // 출금계좌 선택 단계
          Step(
            title: Text('출금계좌 선택', style: AppTextStyles.bodyLarge),
            content: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '자동이체에 사용할 출금계좌를 선택해주세요',
                  style: AppTextStyles.bodyMedium,
                ),
                const SizedBox(height: 16),
                if (wishSelectionState.isLoading)
                  Center(
                    child: Lottie.asset(
                      'assets/images/loading/loading_simple.json',
                      width: 140,
                      height: 140,
                      fit: BoxFit.contain,
                    ),
                  )
                else if (wishSelectionState.accounts.isEmpty)
                  Center(
                    child: Text(
                      '등록된 출금계좌가 없습니다',
                      style: AppTextStyles.bodyMedium,
                    ),
                  )
                else
                  _buildAccountSelection(
                    wishSelectionState.accounts,
                    selectedWithdrawalAccount,
                    (account) => ref.read(selectedWithdrawalAccountProvider.notifier).state = account,
                  ),
              ],
            ),
            isActive: _currentStep >= 1,
          ),
          
          // 입금계좌 선택 단계
          Step(
            title: Text('입금계좌 선택', style: AppTextStyles.bodyLarge),
            content: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '자동이체로 모을 입금계좌를 선택해주세요',
                  style: AppTextStyles.bodyMedium,
                ),
                const SizedBox(height: 16),
                if (wishSelectionState.isLoading)
                  const Center(child: CustomLoadingIndicator())
                else if (wishSelectionState.accounts.isEmpty)
                  Center(
                    child: Text(
                      '등록된 입금계좌가 없습니다',
                      style: AppTextStyles.bodyMedium,
                    ),
                  )
                else
                  _buildAccountSelection(
                    // 출금계좌로 선택된 계좌는 제외
                    wishSelectionState.accounts
                        .where((account) => selectedWithdrawalAccount == null || 
                            account.accountNo != selectedWithdrawalAccount.accountNo)
                        .toList(),
                    selectedDepositAccount,
                    (account) => ref.read(selectedDepositAccountProvider.notifier).state = account,
                  ),
              ],
            ),
            isActive: _currentStep >= 2,
          ),
        ],
        controlsBuilder: (context, details) {
          return Padding(
            padding: const EdgeInsets.only(top: 20),
            child: Row(
              children: [
                if (_currentStep > 0)
                  Expanded(
                    child: OutlinedButton(
                      onPressed: details.onStepCancel,
                      style: OutlinedButton.styleFrom(
                        side: BorderSide(color: AppColors.backgroundBlack),
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: Text(
                        '이전',
                        style: AppTextStyles.bodyMedium.copyWith(
                          color: AppColors.backgroundBlack,
                          fontWeight: FontWeight.w300
                        ),
                      ),
                    ),
                  ),
                if (_currentStep > 0) const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton(
                    onPressed: details.onStepContinue,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.backgroundBlack,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: Text(
                      _currentStep < 2 ? '다음' : '완료',
                      style: AppTextStyles.bodyMedium.copyWith(
                        color: Colors.white,
                        fontWeight: FontWeight.w300
                      ),
                    ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildMonthSelection() {
    final selectedMonth = ref.watch(selectedMonthProvider);
    
    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: _availableMonths.length,
      itemBuilder: (context, index) {
        final month = _availableMonths[index];
        final dailyAmount = _dailyAmounts[index];
        final isSelected = month == selectedMonth;
        
        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: BorderSide(
              color: isSelected ? AppColors.backgroundBlack : AppColors.disabled,
              width: 0.5,
            ),
          ),
          child: InkWell(
            onTap: () => ref.read(selectedMonthProvider.notifier).state = month,
            borderRadius: BorderRadius.circular(12),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  Radio<int>(
                    value: month,
                    groupValue: selectedMonth,
                    onChanged: (value) {
                      if (value != null) {
                        ref.read(selectedMonthProvider.notifier).state = value;
                      }
                    },
                    activeColor: AppColors.backgroundBlack,
                  ),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '$month개월',
                          style: AppTextStyles.bodyLarge.copyWith(
                            fontWeight: FontWeight.w300,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          '일 ${dailyAmount.toString().replaceAllMapped(
                                RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
                                (Match m) => '${m[1]},',
                              )} 원',
                          style: AppTextStyles.bodyMedium.copyWith(
                            color: AppColors.backgroundBlack,
                            fontWeight: FontWeight.w300
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildAccountSelection(
    List<DemDepItem> accounts,
    DemDepItem? selectedAccount,
    Function(DemDepItem) onSelect,
  ) {
    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: accounts.length,
      itemBuilder: (context, index) {
        final account = accounts[index];
        final isSelected = selectedAccount?.accountNo == account.accountNo;
        
        return Card(
          margin: const EdgeInsets.only(bottom: 12),
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: BorderSide(
              color: isSelected ? AppColors.backgroundBlack : AppColors.disabled,
              width: isSelected ? 1 : 0.5,
            ),
          ),
          child: InkWell(
            onTap: () => onSelect(account),
            borderRadius: BorderRadius.circular(12),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  Radio<String>(
                    value: account.accountNo,
                    groupValue: selectedAccount?.accountNo,
                    onChanged: (value) {
                      if (value != null) {
                        onSelect(account);
                      }
                    },
                    activeColor: AppColors.backgroundBlack,
                  ),
                  BankIcon(
                    bankName: account.bankName,
                    size: 30,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          account.bankName,
                          style: AppTextStyles.bodyLarge.copyWith(
                            fontWeight: FontWeight.w400,
                            fontSize: 16
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          _formatAccountNumber(account.accountNo),
                          style: AppTextStyles.bodyMedium.copyWith(
                            fontWeight: FontWeight.w300
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  String _formatAccountNumber(String accountNo) {
    // 계좌번호 포맷 (예: 123-456-789)
    if (accountNo.length < 6) return accountNo;
    
    final midPos = (accountNo.length / 2).round();
    final first = accountNo.substring(0, midPos);
    final last = accountNo.substring(midPos);
    
    return '$first-$last';
  }

  void _registerWishAndAutoTransfer() async {
    final selectedMonth = ref.read(selectedMonthProvider);
    final withdrawalAccount = ref.read(selectedWithdrawalAccountProvider);
    final depositAccount = ref.read(selectedDepositAccountProvider);
    
    if (withdrawalAccount == null || depositAccount == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('출금계좌와 입금계좌를 모두 선택해주세요')),
      );
      return;
    }
    
    // 종료일 계산 (현재로부터 선택한 개월 수만큼)
    final now = DateTime.now();
    final dueDate = DateTime(now.year, now.month + selectedMonth, now.day);
    final dueDateStr = '${dueDate.year}${dueDate.month.toString().padLeft(2, '0')}${dueDate.day.toString().padLeft(2, '0')}';
    
    // 위시 선택 및 자동이체 등록 뷰모델 호출
    try {
      await ref.read(wishSelectionProvider.notifier).selectWishAndCreateAutoTransfer(
        wishlistPk: widget.wishlist.wishlistPk,
        withdrawalAccountNo: withdrawalAccount.accountNo,
        depositAccountNo: depositAccount.accountNo,
        dueDate: dueDateStr,
        transactionBalance: _dailyAmounts[_availableMonths.indexOf(selectedMonth)],  // 일별 금액으로 수정
      );
      
      // 성공 시 완료 페이지로 이동
      if (mounted) {
        context.go(SignupRoutes.getWishCompletePath(), extra: {
          'wishlist': widget.wishlist,
          'selectedMonth': selectedMonth,
          'dailyAmount': _dailyAmounts[_availableMonths.indexOf(selectedMonth)],
          'withdrawalAccount': withdrawalAccount,
          'depositAccount': depositAccount,
          'dueDate': dueDateStr,
        });
      }
    } catch (e) {
      // 에러 처리
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('오류가 발생했습니다: $e')),
        );
      }
    }
  }
}