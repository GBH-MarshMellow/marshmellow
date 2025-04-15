import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/my/salary_model.dart';

class AccountSelectionSection extends StatefulWidget {
  final List<AccountModel> accounts;
  final AccountModel? selectedAccount;
  final Function(AccountModel) onAccountSelected;
  final bool isLoading;
  final String? errorMessage;

  const AccountSelectionSection({
    Key? key,
    required this.accounts,
    required this.selectedAccount,
    required this.onAccountSelected,
    this.isLoading = false,
    this.errorMessage,
  }) : super(key: key);

  @override
  State<AccountSelectionSection> createState() => _AccountSelectionSectionState();
}

class _AccountSelectionSectionState extends State<AccountSelectionSection> {
  bool _isRedirecting = false;
  
  @override
  void didUpdateWidget(AccountSelectionSection oldWidget) {
    super.didUpdateWidget(oldWidget);
    // 계좌 목록이 로드 완료되었고, 비어있으며, 아직 리다이렉트 중이 아닐 때
    if (!widget.isLoading && 
        widget.accounts.isEmpty && 
        widget.errorMessage == null && 
        !_isRedirecting) {
      _redirectToMainPage();
    }
  }

  void _redirectToMainPage() {
    setState(() {
      _isRedirecting = true;
    });
    
    // 3초 후에 메인 페이지로 이동
    Future.delayed(const Duration(seconds: 3), () {
      if (mounted) {
        // 메인 페이지 라우트 경로로 이동
        context.go('/budget'); // 또는 실제 메인 페이지 경로로 변경
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // 로딩 표시
        if (widget.isLoading)
          const Center(
            child: CircularProgressIndicator(),
          )
        // 오류 메시지 표시
        else if (widget.errorMessage != null)
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(
                  Icons.error_outline,
                  color: Colors.red,
                  size: 48,
                ),
                const SizedBox(height: 16),
                Text(
                  widget.errorMessage!,
                  style: AppTextStyles.bodyMedium.copyWith(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: () {
                    // 재시도 로직
                  },
                  child: const Text('다시 시도'),
                ),
              ],
            ),
          )
        // 계좌 목록이 비어있을 때
        else if (widget.accounts.isEmpty)
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(
                  Icons.account_balance_wallet_outlined,
                  color: AppColors.textSecondary,
                  size: 48,
                ),
                const SizedBox(height: 16),
                Text(
                  '계좌 정보가 없습니다.',
                  style: AppTextStyles.bodyMedium.copyWith(color: AppColors.textSecondary),
                ),
                const SizedBox(height: 8),
                // 리다이렉트 중임을 표시하는 메시지
                if (_isRedirecting)
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        '메인 페이지로 이동 중...',
                        style: AppTextStyles.bodySmall.copyWith(color: AppColors.textSecondary),
                      ),
                    ],
                  ),
              ],
            ),
          )
        // 계좌 목록 표시
        else
          ListView.separated(
            physics: const NeverScrollableScrollPhysics(),
            shrinkWrap: true,
            itemCount: widget.accounts.length,
            separatorBuilder: (context, index) => const Divider(height: 1),
            itemBuilder: (context, index) {
              final account = widget.accounts[index];
              final isSelected = widget.selectedAccount?.accountNo == account.accountNo;

              return _buildAccountItem(account, isSelected);
            },
          ),
      ],
    );
  }

  Widget _buildAccountItem(AccountModel account, bool isSelected) {
    return InkWell(
      onTap: () => widget.onAccountSelected(account),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 16.0, horizontal: 8.0),
        child: Row(
          children: [
            // 은행 로고 또는 아이콘
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: AppColors.background,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Center(
                child: Text(
                  account.bankName.substring(0, 1),
                  style: AppTextStyles.bodyLarge,
                ),
              ),
            ),
            const SizedBox(width: 16),
            
            // 계좌 정보
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    account.bankName,
                    style: AppTextStyles.bodySmall.copyWith(
                      color: AppColors.textSecondary,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _formatAccountNumber(account.accountNo),
                    style: AppTextStyles.bodyMedium,
                  ),
                  if (account.accountName.isNotEmpty) ...[
                    const SizedBox(height: 4),
                    Text(
                      account.accountName,
                      style: AppTextStyles.bodySmall.copyWith(
                        color: AppColors.textSecondary,
                      ),
                    ),
                  ],
                ],
              ),
            ),
            
            // 선택 표시
            if (isSelected)
              const Icon(
                Icons.check_circle,
                color: AppColors.backgroundBlack,
                size: 24,
              )
            else
              const Icon(
                Icons.circle_outlined,
                color: AppColors.textSecondary,
                size: 24,
              ),
          ],
        ),
      ),
    );
  }

  // 계좌번호 포맷팅
  String _formatAccountNumber(String accountNo) {
    if (accountNo.length < 8) return accountNo;
    
    final firstPart = accountNo.substring(0, 4);
    final lastPart = accountNo.substring(accountNo.length - 4);
    
    return '$firstPart-****-$lastPart';
  }
}