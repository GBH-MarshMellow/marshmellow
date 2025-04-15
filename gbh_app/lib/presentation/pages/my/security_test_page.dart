// lib/presentation/pages/security/security_test_page.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/config/app_config.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/viewmodels/encryption/encryption_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';

class SecurityTestPage extends ConsumerWidget {
  const SecurityTestPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final aesKeyState = ref.watch(aesKeyNotifierProvider);
    
    return Scaffold(
      appBar: CustomAppbar(
        title: 'Security Test',
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
            Text('Security Test Page', style: AppTextStyles.appBar.copyWith(color: AppColors.pinkPrimary)),
            const SizedBox(height: 30),
            Text('RSA Private Key:'),
            Container(
              height: 100,
              color: Colors.black12,
              child: SingleChildScrollView(
                child: Text(AppConfig.rsaPrivateKey, style: const TextStyle(fontSize: 12)),
              ),
            ),
            const SizedBox(height: 30),
            Row(
              children: [
                Text('AES Key Response:'),
                IconButton(
                  icon: const Icon(Icons.refresh),
                  onPressed: () => ref.read(aesKeyNotifierProvider.notifier).fetchAesKey(),
                ),
              ],
            ),
            aesKeyState.when(
              data: (data) => Container(
                color: AppColors.yellowDark,
                width: MediaQuery.of(context).size.width,
                child: SelectableText(data.toString()),
              ),
              loading: () => const CircularProgressIndicator(),
              error: (error, stack) => Text('에러: $error'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () async {
                final storedKey = await ref.read(aesKeyNotifierProvider.notifier).getStoredAesKey();
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('저장된 AES 키: ${storedKey ?? "없음"}')),
                );
              },
              child: Text('저장된 AES 키 확인'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () async {
                try {
                  // 테스트용 암호화/복호화
                  final encryptionService = ref.read(encryptionServiceProvider);
                  final storedKey = await encryptionService.getAesKey();
                  
                  if (storedKey != null) {
                    final testText = '안녕하세요, 암호화 테스트입니다.';
                    final encrypted = await encryptionService.encryptWithAes(testText);
                    final decrypted = await encryptionService.decryptWithAes(encrypted);
                    
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('복호화 결과: $decrypted')),
                    );
                  } else {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('AES 키가 없습니다. 먼저 키를 가져와주세요.')),
                    );
                  }
                } catch (e) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('암호화/복호화 테스트 실패: $e')),
                  );
                }
              },
              child: Text('암호화/복호화 테스트'),
            ),
            const SizedBox(height: 30),
          ],
        ),
      ),
    );
  }
}