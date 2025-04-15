// lib/presentation/pages/finance/finance_test_page.dart
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart'; 
import 'package:marshmellow/core/config/app_config.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';
import 'package:marshmellow/router/routes/budget_routes.dart';
import 'package:marshmellow/router/routes/finance_routes.dart'; // 경로 상수 import

// 로딩인디케이터 추가
import 'package:marshmellow/presentation/widgets/loading/loading_manager.dart';
import 'package:marshmellow/router/routes/my_routes.dart';

class FinanceTestPage extends StatelessWidget {
  const FinanceTestPage({Key? key}) : super(key: key);

  // 가상의 API 호출 함수 (0.1초 소요)
  Future<String> _mockFastApiCall() async {
    // 0.1초 대기
    await Future.delayed(const Duration(milliseconds: 100));
    return "API 응답 데이터";
  }

// >>>>>>>>>>>>>>>>>>>>>>> API호출시 로딩 인디케이터 >>>>>>>>>>>>>>>>>>>>>>>>>>
  Future<void> _testApiWithLoading(BuildContext context) async {
    // ☆★ API호출 시작하자마자 로딩 시작 ☆★
    LoadingManager.show(
      context,
      text: "API 호출 중...",
      backgroundColor: Colors.purpleAccent,
      opacity: 0.2,
      durationInSeconds: 0, // ☆★ 0하면, 자동 종료 없음. 즉 (수동종료코드)필요함 ☆★
      minimumDurationInSeconds: 1,
    );
    try {
      final result = await _mockFastApiCall(); // 가상 API호출
      print("API 결과: $result");
    } catch (error) {
      print("API 오류: $error");
    } finally {
      
      await LoadingManager.hide(); // ☆★ 완료시 로딩 종료시켜주기 (수동종료코드) ☆★
      
      // 테스트용 스낵바
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("API 호출 완료! (0.1초 걸렸지만 로딩은 1초 표시)"),
          duration: Duration(seconds: 2),
        ),
      );
    }
  }
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< API호출시 로딩 인디케이터 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppbar(
        title: 'Test Page',
        actions: [
          IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: () {
                context.pushReplacement(FinanceRoutes.getTestPath());
              },
            ),
        ],
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  context.pop(); // 이전 페이지로 돌아가기
                },
                child: const Text('돌아가기'),
              ),
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<< 키보드 테스트 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  // 테스트 페이지로 이동
                  context.push(FinanceRoutes.getKeyboardTestPath());
                },
                child: const Text('키보드 테스트'),
              ),
        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>> 키보드 테스트 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<< 로딩 인디케이터 테스트 추가 <<<<<<<<<<<<<<<
              // 로딩 인디케이터 테스트 버튼 추가
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  // 로딩인디케이터 로직 
                  LoadingManager.show(
                    context, // 필수
                    text: "사용자 정보를 불러오는 중...", // 메세지
                    backgroundColor: Colors.blue, // 배경색 (기본검정)
                    opacity: 0.6, // 투명도 (기본0.7)
                    durationInSeconds: 3, // 3초 후 종료 (기본 1초)
                    minimumDurationInSeconds: 1, // 최소 1초 동안 표시 (기본 1초)
                  );
                },
                child: const Text('2초 로딩 테스트'),
              ),
        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>> 로딩 인디케이터 테스트 추가 >>>>>>>>>>>>>>>>
        
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<< API호출시 로딩 인디케이터  <<<<<<<<<<<<<<<
              // 가상 API 호출 테스트 버튼 (새로 추가)
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () => _testApiWithLoading(context),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                ),
                child: const Text('0.1초 API 호출 테스트 (1초 로딩)'),
              ),
        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>> API호출시 로딩 인디케이터 >>>>>>>>>>>>>>>>
        
              const SizedBox(height: 20), // 남기 datepicker 테스트페이지
              ElevatedButton(
                onPressed: () {
                  // GoRouter를 사용하여 테스트 페이지로 이동
                  context.push(MyRoutes.getDatepickerTestPath());
                },
                child: const Text('데이트피커 테스트페이지'),
              ),
              ElevatedButton(
                onPressed: () {
                  // GoRouter를 사용하여 테스트 페이지로 이동
                  context.push(SignupRoutes.root);
                },
                child: const Text('회원가입 테스트페이지'),
              ),
              ElevatedButton(
                onPressed: () {
                  // GoRouter를 사용하여 테스트 페이지로 이동
                  context.push(SignupRoutes.getMyDataSplashPath());
                },
                child: const Text('인증서 테스트페이지'),
              ),
              ElevatedButton(
                onPressed: () {
                  // GoRouter를 사용하여 테스트 페이지로 이동
                  context.push(SignupRoutes.getBudgetCelebratePath());
                },
                child: const Text('월급날 테스트페이지'),
              ),
              ElevatedButton(
                onPressed: () {
                  // GoRouter를 사용하여 테스트 페이지로 이동
                  context.push(SignupRoutes.getWishCreatePath());
                },
                child: const Text('위시 등록 테스트페이지'),
              ),
              ElevatedButton(
                onPressed: () {
                  // GoRouter를 사용하여 테스트 페이지로 이동
                  context.push(SignupRoutes.getSalaryInputPath());
                },
                child: const Text('월급정보 입력 테스트페이지'),
              ),
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<< 로딩 인디케이터 테스트  <<<<<<<<<<<<<<<<<<
              const SizedBox(height: 20),
        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>> 암호화테스트 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
              ElevatedButton(
                onPressed: () {
                  context.push(MyRoutes.getSecurityTestPath());
                },
                child: const Text('암호화 테스트페이지'),
              ),
              ElevatedButton(
                onPressed: () {
                  context.push(MyRoutes.getEncryptionTestPath());
                },
                child: const Text('자산 암호화 테스트페이지'),
              ),
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<< 암호화테스트 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            ],
          ),
        ),
      ),
    );
  }
}