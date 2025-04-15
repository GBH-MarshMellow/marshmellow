// presentation/services/transfer_service.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/certification_select_content.dart';
import 'package:marshmellow/presentation/viewmodels/finance/transfer_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/finance/withdrawal_account_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_info_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_secure_info_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/finance/certificate_login_modal.dart';
import 'package:marshmellow/presentation/widgets/loading/loading_manager.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';

class TransferService {
  // 송금 버튼 클릭 핸들러
  
  static Future<void> handleTransfer(BuildContext context, WidgetRef ref, String accountNo, String bankName) async {
    // 로딩 표시
    final userInfoState = ref.watch(userInfoProvider);
    final userSecureInfoState = ref.watch(userSecureInfoProvider);
    LoadingManager.show(context, text: '계좌 확인 중...', opacity: 1.0, backgroundColor: AppColors.background);
    
    try {
      // 출금계좌 등록 여부 확인
      final withdrawalViewModel = ref.read(withdrawalAccountProvider.notifier);
      final result = await withdrawalViewModel.isAccountRegisteredAsWithdrawal(accountNo);
      final isRegistered = result['isRegistered'] as bool;
      final withdrawalAccountId = result['withdrawalAccountId'] as int?;
      
      // 로딩 숨기기
      LoadingManager.hide();
      print('@@출출@@ : ${result} + ${isRegistered} + ${withdrawalAccountId}');
      
      if (isRegistered && withdrawalAccountId != null) {
        // 이미 등록된 출금계좌라면 2초 후 인증서 로그인 모달 표시
        Future.delayed(const Duration(seconds: 1), () {
          if (context.mounted) {
            // showCertificateLoginModal(
            //   context, 
            //   accountNo: accountNo,
            //   withdrawalAccountId: withdrawalAccountId,
            // );
            showCertificateModal(
              context: context, 
              ref: ref, 
              // userName: '임남기', 
              userName: '${userSecureInfoState.userName ?? '사용자'}', 
              expiryDate: '2028.03.14.', 
              onConfirm: () {
                // TODO: 여기서 인증서 확인 작업 필요
                // 서버에 개인키해싱값 전달 응답 받기
                // 모달 닫고 인증 페이지로 이동
                // Navigator.pop(context);
                ref.read(transferProvider.notifier).setWithdrawalBankName(bankName);
                context.push(
                  FinanceRoutes.getTransferPath(), 
                  extra: {'accountNo': accountNo, 'withdrawalAccountId': withdrawalAccountId,'bankName': bankName}
                );
              }
            );
          }
        });
      } else {
        // 등록되지 않은 계좌라면 출금계좌 등록 페이지로 이동
        if (context.mounted) {
          context.push(FinanceRoutes.getWithdrawalAccountRegistrationPath(accountNo));
        }
      }
    } catch (e) {
      // 오류 발생 시 로딩 숨기고 오류 메시지 표시
      LoadingManager.hide();
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('계좌 확인 중 오류가 발생했습니다: ${e.toString()}')),
        );
      }
    }
  }
}