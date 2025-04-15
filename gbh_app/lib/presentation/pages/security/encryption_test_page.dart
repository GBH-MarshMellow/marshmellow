// lib/presentation/pages/security/encryption_test_page.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/datasources/remote/finance_api.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'package:marshmellow/presentation/viewmodels/encryption/encryption_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/core/utils/encryption_util.dart';
import 'package:marshmellow/di/providers/core_providers.dart';

// FinanceAPI 프로바이더 (이미 있으면 기존 것 사용)
final financeApiProvider = Provider<FinanceApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return FinanceApi(apiClient);
});

class EncryptionTestPage extends ConsumerWidget {
  const EncryptionTestPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final aesKeyState = ref.watch(aesKeyNotifierProvider);
    
    return Scaffold(
      appBar: CustomAppbar(
        title: '암호화 API 테스트',
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.read(aesKeyNotifierProvider.notifier).fetchAesKey(),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('암호화 API 테스트 페이지', style: AppTextStyles.appBar.copyWith(color: AppColors.pinkPrimary)),
            const SizedBox(height: 20),
            
            // AES 키 상태 표시
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('AES 키 상태:'),
                aesKeyState.when(
                  data: (data) => data.isNotEmpty 
                      ? Text('준비됨', style: TextStyle(color: Colors.green))
                      : Text('없음', style: TextStyle(color: Colors.red)),
                  loading: () => CircularProgressIndicator(),
                  error: (e, _) => Text('오류: $e', style: TextStyle(color: Colors.red)),
                ),
              ],
            ),
            const SizedBox(height: 10),
            
            // AES 키 가져오기 버튼
            ElevatedButton(
              onPressed: () => ref.read(aesKeyNotifierProvider.notifier).fetchAesKey(),
              child: Text('AES 키 가져오기'),
            ),
            const SizedBox(height: 20),
            
            // 전체 자산 정보 가져오기 테스트
            ElevatedButton(
              onPressed: () async {
                try {
                  // AES 키 확인
                  final aesKey = await ref.read(encryptionServiceProvider).getAesKey();
                  if (aesKey == null || aesKey.isEmpty) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('AES 키가 없습니다. 먼저 키를 가져와주세요.')),
                    );
                    return;
                  }
                  
                  // 자산 정보 API 호출
                  final financeApi = ref.read(financeApiProvider);
                  final result = await financeApi.getAssetInfo();
                  
                  // 결과 출력
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('자산 정보 가져오기 성공!')),
                  );
                  
                  // 상세 정보 다이얼로그 표시
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: Text('자산 정보 결과'),
                      content: Container(
                        width: double.maxFinite,
                        height: 300,
                        child: SingleChildScrollView(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('카드 총액: ${result.data?.cardData?.totalAmount ?? "N/A"}'),
                              Text('예금 총액: ${result.data?.depositData?.totalAmount ?? "N/A"}'),
                              Text('적금 총액: ${result.data?.savingsData?.totalAmount ?? "N/A"}'),
                              Text('입출금 총액: ${result.data?.demandDepositData?.totalAmount ?? "N/A"}'),
                              const SizedBox(height: 20),
                              if (result.data?.cardData?.cardList?.isNotEmpty ?? false)
                                Text('카드 목록:', style: TextStyle(fontWeight: FontWeight.bold)),
                              ...(result.data?.cardData?.cardList ?? []).map((card) => 
                                Padding(
                                  padding: const EdgeInsets.only(left: 16.0, top: 8.0),
                                  child: Text('${card.cardName}: ${card.cardBalance ?? "0"}'),
                                )
                              ),
                            ],
                          ),
                        ),
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.of(context).pop(),
                          child: Text('닫기'),
                        ),
                      ],
                    ),
                  );
                  
                } catch (e) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('자산 정보 가져오기 실패: $e')),
                  );
                }
              },
              child: Text('전체 자산 정보 가져오기'),
            ),
            const SizedBox(height: 10),
            
            // 입출금 계좌 내역 테스트
            ElevatedButton(
              onPressed: () async {
                try {
                  // AES 키 확인
                  final aesKey = await ref.read(encryptionServiceProvider).getAesKey();
                  if (aesKey == null || aesKey.isEmpty) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('AES 키가 없습니다. 먼저 키를 가져와주세요.')),
                    );
                    return;
                  }
                  
                  // 먼저 계좌 목록 가져오기
                  final financeApi = ref.read(financeApiProvider);
                  final assetInfo = await financeApi.getAssetInfo();
                  
                  // 입출금 계좌가 있는지 확인
                  final accountList = assetInfo.data?.demandDepositData?.demandDepositList;
                  if (accountList == null || accountList.isEmpty) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('입출금 계좌가 없습니다.')),
                    );
                    return;
                  }
                  
                  // 첫 번째 계좌의 내역 조회
                  final firstAccount = accountList.first;
                  final transactions = await financeApi.getDemandAccountTransactions(
                    accountNo: firstAccount.accountNo!,
                    startDate: '20250301',
                    endDate: '20250330',
                    transactionType: 'A',
                    orderByType: 'DESC',
                  );
                  
                  // 결과 출력
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('입출금 내역 조회 성공!')),
                  );
                  
                  // 상세 정보 다이얼로그 표시
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: Text('입출금 내역'),
                      content: Container(
                        width: double.maxFinite,
                        height: 300,
                        child: SingleChildScrollView(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('계좌번호: ${firstAccount.accountNo}'),
                              Text('계좌명: ${firstAccount.accountName}'),
                              const SizedBox(height: 10),
                              Text('거래 내역 수: ${transactions.data?.transactionList?.length ?? 0}'),
                              const SizedBox(height: 10),
                              ...(transactions.data?.transactionList ?? []).map((tx) => 
                                Padding(
                                  padding: const EdgeInsets.only(left: 8.0, top: 8.0),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Text('날짜: ${tx.transactionDate}'),
                                      Text('금액: ${tx.transactionBalance}'),
                                      Text('잔액: ${tx.transactionAfterBalance}'),
                                      Text('내용: ${tx.transactionSummary}'),
                                      Divider(),
                                    ],
                                  ),
                                )
                              ),
                            ],
                          ),
                        ),
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.of(context).pop(),
                          child: Text('닫기'),
                        ),
                      ],
                    ),
                  );
                  
                } catch (e) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('입출금 내역 조회 실패: $e')),
                  );
                }
              },
              child: Text('입출금 계좌 내역 조회'),
            ),
            
            const SizedBox(height: 20),
            
            // 암호화 디버그 정보
            Text('암호화 디버그 정보:', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            ElevatedButton(
              onPressed: () async {
                try {
                  // 암호화 유틸리티
                  final encryptionUtil = ref.read(encryptionUtilProvider);
                  
                  // 샘플 요청 데이터
                  final sampleRequest = {
                    'accountNo': '1234567890',
                    'startDate': '20250301',
                    'endDate': '20250330',
                  };
                  
                  // 요청 암호화 테스트
                  final encryptedRequest = await encryptionUtil.encryptRequest(sampleRequest);
                  
                  // 결과 출력
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: Text('암호화 테스트 결과'),
                      content: Container(
                        width: double.maxFinite,
                        height: 300,
                        child: SingleChildScrollView(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('원본 요청:', style: TextStyle(fontWeight: FontWeight.bold)),
                              Text('accountNo: ${sampleRequest['accountNo']}'),
                              Text('startDate: ${sampleRequest['startDate']}'),
                              Text('endDate: ${sampleRequest['endDate']}'),
                              
                              const SizedBox(height: 20),
                              Text('암호화된 요청:', style: TextStyle(fontWeight: FontWeight.bold)),
                              Text('IV: ${encryptedRequest['iv']}'),
                              Text('accountNo: ${encryptedRequest['accountNo']}'),
                              Text('startDate: ${encryptedRequest['startDate']}'),
                              Text('endDate: ${encryptedRequest['endDate']}'),
                            ],
                          ),
                        ),
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.of(context).pop(),
                          child: Text('닫기'),
                        ),
                      ],
                    ),
                  );
                } catch (e) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('암호화 테스트 실패: $e')),
                  );
                }
              },
              child: Text('암호화 테스트'),
            ),
          ],
        ),
      ),
    );
  }
}