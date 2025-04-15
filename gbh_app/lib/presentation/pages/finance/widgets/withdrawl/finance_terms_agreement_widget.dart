// finance_terms_agreement_widget.dart (새 파일)
import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/viewmodels/finance/withdrawal_account_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';

class FinanceTermsAgreementWidget extends StatelessWidget {
  final WithdrawalAccountState state;
  final WithdrawalAccountViewModel viewModel;

  const FinanceTermsAgreementWidget({
    Key? key,
    required this.state,
    required this.viewModel,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SizedBox(height: 40,),
          Text('서비스를 이용하려면' , style: AppTextStyles.appBar.copyWith(fontWeight: FontWeight.w600,),),
          SizedBox(height: 8,),
          Text('동의가 필요해요',  style: AppTextStyles.appBar.copyWith(fontWeight: FontWeight.w600,)),
          SizedBox(height: 40,),
          // 약관 전체 동의 체크박스 (회색 배경으로 강조)
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                children: [
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.grey[200],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: CheckboxListTile(
                      title: const Text('약관 전체 동의', style: AppTextStyles.bodyMedium),
                      value: state.isAllTermsAgreed,
                      onChanged: (_) => viewModel.toggleAllTermsAgreement(),
                      controlAffinity: ListTileControlAffinity.leading,
                      // contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    ),
                  ),
              
                  const SizedBox(height: 36),
                  
                  // 오픈뱅킹 약관 필수동의 (펼치기 가능)
                  _buildExpandableTermsItem(
                    title: '오픈뱅킹 약관 필수동의',
                    isExpanded: state.isFirstTermExpanded,
                    isChecked: state.isFirstTermAgreed,
                    onCheckChanged: (_) => viewModel.toggleFirstTermAgreement(),
                    onExpandChanged: () => viewModel.toggleFirstTermExpanded(),
                    children: [
                      // 하위 약관들
                      _buildSubTermsItem(
                        title: '오픈뱅킹 서비스 이용약관',
                        isChecked: state.isFirstTermAgreed, // 상위 약관과 동일한 상태 사용
                        onCheckChanged: (_) => viewModel.toggleFirstTermAgreement(),
                        onNavigate: () {
                          context.push(FinanceRoutes.getAgreementPath('A001'));
                        },
                      ),
                      _buildSubTermsItem(
                        title: '고객본인확인',
                        isChecked: state.isFirstTermAgreed, // 상위 약관과 동일한 상태 사용
                        onCheckChanged: (_) => viewModel.toggleFirstTermAgreement(),
                        onNavigate: () {
                          context.push(FinanceRoutes.getAgreementPath('A002'));
                        },
                      ),
                    ],
                  ),
                  
                  const SizedBox(height: 12),
                  
                  // 개인(신용)정보 및 금융거래정보 필수동의
                  _buildExpandableTermsItem(
                    title: '개인(신용)정보 및 금융거래정보 필수동의',
                    isExpanded: state.isSecondTermExpanded,
                    isChecked: state.isSecondTermAgreed,
                    onCheckChanged: (_) => viewModel.toggleSecondTermAgreement(),
                    onExpandChanged: () => viewModel.toggleSecondTermExpanded(),
                    children: [
                      // 하위 약관들
                      _buildSubTermsItem(
                        title: '오픈뱅킹용 개인(신용)정보 수집.이용',
                        isChecked: state.isSecondTermAgreed,
                        onCheckChanged: (_) => viewModel.toggleSecondTermAgreement(),
                        onNavigate: () {
                          context.push(FinanceRoutes.getAgreementPath('B001'));
                        },
                      ),
                      _buildSubTermsItem(
                        title: '오픈뱅킹용 개인(신용)정보 제공',
                        isChecked: state.isSecondTermAgreed,
                        onCheckChanged: (_) => viewModel.toggleSecondTermAgreement(),
                        onNavigate: () {
                          context.push(FinanceRoutes.getAgreementPath('B002'));
                        },
                      ),
                      _buildSubTermsItem(
                        title: '오픈뱅킹용 금융거래정보 제공',
                        isChecked: state.isSecondTermAgreed,
                        onCheckChanged: (_) => viewModel.toggleSecondTermAgreement(),
                        onNavigate: () {
                          context.push(FinanceRoutes.getAgreementPath('B003'));
                        },
                      ),
                      _buildSubTermsItem(
                        title: '오픈뱅킹용 기기정보 수집.이용',
                        isChecked: state.isSecondTermAgreed,
                        onCheckChanged: (_) => viewModel.toggleSecondTermAgreement(),
                        onNavigate: () {
                          context.push(FinanceRoutes.getAgreementPath('B004'));
                        },
                      ),
                      _buildSubTermsItem(
                        title: '오픈뱅킹용 기기정보 제공',
                        isChecked: state.isSecondTermAgreed,
                        onCheckChanged: (_) => viewModel.toggleSecondTermAgreement(),
                        onNavigate: () {
                          context.push(FinanceRoutes.getAgreementPath('B005'));
                        },
                      ),
                    ],
                  ),
                  
                  const SizedBox(height: 12),
                  
                  // 개인(신용)정보 수집.이용 선택동의
                  _buildExpandableTermsItem(
                    title: '개인(신용)정보 수집.이용 선택동의',
                    isExpanded: state.isThirdTermExpanded,
                    isChecked: state.isThirdTermAgreed,
                    onCheckChanged: (_) => viewModel.toggleThirdTermAgreement(),
                    onExpandChanged: () => viewModel.toggleThirdTermExpanded(),
                    children: [
                      // 하위 약관
                      _buildSubTermsItem(
                        title: '회원의 신용도 평가',
                        isChecked: state.isThirdTermAgreed,
                        onCheckChanged: (_) => viewModel.toggleThirdTermAgreement(),
                        onNavigate: () {
                          context.push(FinanceRoutes.getAgreementPath('C001'));
                        },
                      ),
                    ],
                  ),
                  
                  SizedBox(height: 20),
                ],
              ),
            ),
          ),
          
          // 확인 버튼
          Button(
            text: '확인',
            onPressed: _canProceed(state) ? () => viewModel.sendVerificationCode() : null,
          ),
          SizedBox(height: 30,)
        ],
      ),
    );
  }

  // 하위 약관 아이템 위젯
  Widget _buildSubTermsItem({
    required String title,
    required bool isChecked,
    required Function(bool?) onCheckChanged,
    required VoidCallback onNavigate,
  }) {
    return Padding(
      padding: const EdgeInsets.only(left: 32),
      child: Row(
        children: [
          // 체크박스
        _buildRoundCheckbox(
          value: isChecked,
          onChanged: onCheckChanged,
          activeColor: Colors.black,  // 원하는 색상으로 변경
          size: 20,
        ),
          const SizedBox(width: 8),
          // 제목
          Expanded(
            child: Text(title, style: AppTextStyles.bodySmall),
          ),
          // 상세보기 아이콘 버튼
          IconButton(
            icon: const Icon(Icons.keyboard_arrow_right, size: 20),
            onPressed: onNavigate,
          ),
        ],
      ),
    );
  }

  // 확장 가능한 약관 아이템 위젯
  Widget _buildExpandableTermsItem({
    required String title,
    required bool isExpanded,
    required bool isChecked,
    required Function(bool?) onCheckChanged,
    required VoidCallback onExpandChanged,
    required List<Widget> children,
  }) {
    return Column(
      children: [
        Row(
          children: [
        SizedBox(width: 16,),
            // 체크박스
        _buildRoundCheckbox(
          value: isChecked,
          onChanged: onCheckChanged,
          activeColor: Colors.black,  // 원하는 색상으로 변경
          size: 20,
        ),
        SizedBox(width: 10,),
            // 제목
            Expanded(
              child: Text(title , style: AppTextStyles.bodyMedium.copyWith(fontSize: 14)),
            ),
            // 펼치기/접기 아이콘 버튼
            IconButton(
              icon: Icon(isExpanded ? Icons.keyboard_arrow_down : Icons.keyboard_arrow_right),
              onPressed: onExpandChanged,
            ),
          ],
        ),
        // 펼쳐진 내용
        if (isExpanded) ...children,
      ],
    );
  }

  // 다음 단계로 진행 가능한지 확인하는 함수
  bool _canProceed(WithdrawalAccountState state) {
    // 필수 약관 동의 확인 (선택 동의는 제외)
    return state.isFirstTermAgreed && state.isSecondTermAgreed;
  }
}

// 동그란 체크박스 커스텀 위젯
Widget _buildRoundCheckbox({
  required bool value,
  required Function(bool?) onChanged,
  Color activeColor = Colors.green,  // 체크됐을 때 색상
  Color borderColor = Colors.grey,   // 테두리 색상
  double size = 24.0,
}) {
  return InkWell(
    onTap: () => onChanged(!value),
    borderRadius: BorderRadius.circular(size / 2),
    child: Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        border: Border.all(
          color: value ? activeColor : borderColor,
          width: 1.5,
        ),
        color: value ? activeColor : Colors.transparent,
      ),
      child: value
          ? Icon(
              Icons.check,
              size: size * 0.75,
              color: Colors.white,
            )
          : null,
    ),
  );
}