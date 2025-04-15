import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/custom_button.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/salary_input/account_selection_section.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/salary_input/deposit_selection_section.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/salary_input/salary_registration_section.dart';
import 'package:marshmellow/presentation/viewmodels/my/salary_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_info_viewmodel.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';


final salaryInputStepProvider = StateProvider<int>((ref) => 0);

class SalaryInputPage extends ConsumerStatefulWidget{
  const SalaryInputPage({Key? key}) : super(key: key);

  @override
  _SalaryInputPageState createState() => _SalaryInputPageState();
}

class _SalaryInputPageState extends ConsumerState<SalaryInputPage> {

  @override
  void initState() {
    super.initState();

    // 페이지 로드 시 계좌 목록 조회
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref.read(mySalaryProvider.notifier).fetchAccounts();
    });
  }

  @override
  Widget build(BuildContext context) {
    // 현재 월급 입력 단계
    final currentStep = ref.watch(salaryInputStepProvider);
    // 월급 관련 상태
    final salaryState = ref.watch(mySalaryProvider);
    // 사용자 정보 관련 상태
    final userInfoState = ref.watch(userInfoProvider);

    return Scaffold(
      appBar: _buildAppBar(context, currentStep),
      body: Padding(
        padding: const EdgeInsets.all(40.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: 40),
            Text(_getStepTitle(currentStep), style: AppTextStyles.mainTitle),
            const SizedBox(height: 20),
            Text(
              _getStepDescription(currentStep),
              style: AppTextStyles.bodyMedium.copyWith(color: AppColors.textSecondary),
            ),
            const SizedBox(height: 40),

            // 단계별 컨텐츠
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    // 1단계: 계좌 선택
                    if (currentStep == 0)
                      AccountSelectionSection(
                        accounts: salaryState.accounts,
                        selectedAccount: salaryState.selectedAccount,
                        onAccountSelected: (account) {
                          ref.read(mySalaryProvider.notifier).selectAccount(account);
                        },
                        isLoading: salaryState.isLoading,
                        errorMessage: salaryState.errorMessage,
                      ),

                    // 2단계: 입금 내역 선택
                    if (currentStep == 1)
                      DepositSelectionSection(
                        deposits: salaryState.deposits,
                        selectedDeposit: salaryState.selectedDeposit,
                        onDepositSelected: (deposit) {
                          ref.read(mySalaryProvider.notifier).selectDeposit(deposit);
                        },
                        isLoading: salaryState.isLoading,
                        errorMessage: salaryState.errorMessage,
                      ),

                    // 3단계: 월급 등록 확인
                    if (currentStep == 2)
                      SalaryRegistrationSection(
                        account: salaryState.selectedAccount!,
                        deposit: salaryState.selectedDeposit!,
                        isLoading: userInfoState.isLoading,
                        errorMessage: userInfoState.error,
                      ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 20),

            // 하단 버튼 영역
            _buildBottomButton(context, currentStep, salaryState),
          ],
        ),
      ),
    );
  }

  // 앱바 구성
  AppBar _buildAppBar(BuildContext context, int currentStep) {
    return AppBar(
      backgroundColor: Colors.transparent,
      elevation: 0,
      automaticallyImplyLeading: false,
      title: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          _buildStepIndicator(0, currentStep >= 0),
          _buildStepConnector(),
          _buildStepIndicator(1, currentStep >= 1),
          _buildStepConnector(),
          _buildStepIndicator(2, currentStep >= 2),
        ],
      ),
      centerTitle: true,
    );
  }

  // 단계 표시 원형 위젯
  Widget _buildStepIndicator(int step, bool isActive) {
    return Container(
      width: 24,
      height: 24,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: isActive ? AppColors.pinkPrimary : AppColors.disabled,
      ),
      child: Center(
        child: Text(
          (step + 1).toString(),
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 12,
          ),
        ),
      ),
    );
  }

  // 단계 연결선
  Widget _buildStepConnector() {
    return Container(
      width: 20,
      height: 1,
      color: AppColors.disabled,
    );
  }

  // 각 단계별 제목
  String _getStepTitle(int step) {
    switch (step) {
      case 0:
        return '계좌 선택';
      case 1:
        return '입금 내역 선택';
      case 2:
        return '월급 등록';
      default:
        return '';
    }
  }

  // 각 단계별 설명
  String _getStepDescription(int step) {
    switch (step) {
      case 0:
        return '월급이 입금되는 계좌를 선택해주세요.';
      case 1:
        return '월급으로 의심되는 입금 내역을 선택해주세요.';
      case 2:
        return '선택한 정보로 월급을 등록합니다.';
      default:
        return '';
    }
  }

  // 하단 버튼
  Widget _buildBottomButton(BuildContext context, int currentStep, SalaryState salaryState) {
    // 단계별 버튼 활성화 조건 및 텍스트
    bool isButtonEnabled = false;
    String buttonText = '다음';
    
    switch (currentStep) {
      case 0:
        isButtonEnabled = salaryState.selectedAccount != null;
        break;
      case 1:
        isButtonEnabled = salaryState.selectedDeposit != null;
        break;
      case 2:
        isButtonEnabled = true;
        buttonText = '완료';
        break;
    }

    return CustomButton(
      text: buttonText,
      onPressed: isButtonEnabled ? () => _handleButtonPress(currentStep) : null,
      isEnabled: isButtonEnabled,
    );
  }

  // 버튼 클릭 처리
  void _handleButtonPress(int currentStep) async {
    switch (currentStep) {
      case 0:
        // 계좌 선택 완료 -> 입금 내역 조회
        await ref.read(mySalaryProvider.notifier).fetchDeposits();
        // 다음 단계로 이동
        ref.read(salaryInputStepProvider.notifier).state++;
        break;
        
      case 1:
        // 입금 내역 선택 완료 -> 다음 단계로 이동
        ref.read(salaryInputStepProvider.notifier).state++;
        break;
        
      case 2:
        // 월급 등록 처리
        final salaryState = ref.read(mySalaryProvider);
        
        if (salaryState.selectedAccount != null && salaryState.selectedDeposit != null) {
          // 날짜에서 일자만 추출 (예: "20250328" -> 28)
          final dateStr = salaryState.selectedDeposit!.transactionDate;
          final day = int.parse(dateStr.substring(6, 8));
          
          // 월급 등록 API 호출
          await ref.read(userInfoProvider.notifier).myRegisterSalary(
            salaryState.selectedDeposit!.transactionBalance,
            day,
            salaryState.selectedAccount!.accountNo,
          );
          
          // 등록 완료 후 다음 화면으로 이동
          if (!context.mounted) return;
          
          // 월급 정보 입력 완료 확인 페이지로 이동
          context.go(SignupRoutes.getSalaryInputCompletePath());
        }
        break;
    }
  }
}