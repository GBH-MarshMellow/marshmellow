import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';

// 인증서 로그인 모달 표시 함수
void showCertificateLoginModal(BuildContext context, {required String accountNo, required int withdrawalAccountId,}) {
  showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
    ),
    builder: (BuildContext context) {
      return Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              '인증서 로그인',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            const Text(
              '출금을 위해 인증서 로그인이 필요합니다.',
              style: TextStyle(fontSize: 14),
            ),
            const SizedBox(height: 24),
            
            // 인증서 목록 (임시로 하나만 표시)
            GestureDetector(
              onTap: () {
                // 모달 닫고 인증 페이지로 이동
                Navigator.pop(context);
                context.push(
                  FinanceRoutes.getAuthPath(), 
                  extra: {'accountNo': accountNo, 'withdrawalAccountId': withdrawalAccountId,}
                );
              },
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade300),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(Icons.verified_user, color: Colors.blue.shade700),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '금융인증서',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            '유효기간: 2025.01.01',
                            style: TextStyle(fontSize: 12, color: Colors.grey),
                          ),
                        ],
                      ),
                    ),
                    const Icon(Icons.arrow_forward_ios, size: 16),
                  ],
                ),
              ),
            ),
          ],
        ),
      );
    },
  );
}