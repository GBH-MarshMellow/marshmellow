import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart'; 

class FinanceTransferPage extends StatelessWidget {
  const FinanceTransferPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('transfer page'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text('송금페이지 '),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                context.pop(); // 이전 페이지로 돌아가기
              },
              child: const Text('돌아가기'),
            ),
          ],
        
        ),
      ),
    );
  }
} 