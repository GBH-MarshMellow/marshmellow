import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/viewmodels/finance/transfer_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/dots_input/dots_input.dart';
import 'package:marshmellow/presentation/widgets/keyboard/index.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';

class CertificateAuthPage extends ConsumerStatefulWidget {
  final String accountNo;
  final int withdrawalAccountId;

  const CertificateAuthPage({
    Key? key,
    required this.accountNo,
    required this.withdrawalAccountId, // 생성자 매개변수 추가
  }) : super(key: key);

  @override
  ConsumerState<CertificateAuthPage> createState() => _CertificateAuthPageState();
}

class _CertificateAuthPageState extends ConsumerState<CertificateAuthPage> {
  final TextEditingController _pinController = TextEditingController();
  String _pinValue = '';
  int _wrongAttempts = 0;
  bool _isLoading = false;

  // 임시 비밀번호 (실제로는 서버에서 검증)
  static const String _tempCorrectPin = '123456';

  @override
  void dispose() {
    _pinController.dispose();
    super.dispose();
  }

@override
void initState() {
  super.initState();
  
  // 페이지 로드 후 약간의 딜레이를 주고 키보드 표시
  WidgetsBinding.instance.addPostFrameCallback((_) {
    _showKeyboard();
  });
}

// 키보드 표시 함수 (기존의 onTap 콜백 내용을 분리)
void _showKeyboard() async {
  setState(() {
    _pinValue = '';
    _pinController.text = '';
  });
  
  await KeyboardModal.showSecureNumericKeyboard(
    context: context,
    onValueChanged: (value) {
      setState(() {
        _pinValue = value;
        _pinController.text = '•' * value.length;
      });
    },
    initialValue: '',
    maxLength: 6, // 6자리 핀번호
    obscureText: true,
  );
  
  // PIN 입력 완료 후 자동 검증
  if (_pinValue.length == 6) {
    _verifyPin();
  }
}

  // PIN 입력 및 검증
  // _verifyPin 메서드 수정
  void _verifyPin() async {
    setState(() {
      _isLoading = true;
    });

    // 실제 앱에서는 API 호출로 대체
    await Future.delayed(const Duration(seconds: 1));

    if (_pinValue == _tempCorrectPin) {
      // 인증 성공
      setState(() {
        _isLoading = false;
      });
      
      if (mounted) {
        // TransferViewModel에서 송금 실행 후 결과에 따라 완료 페이지로 이동
        final transferViewModel = ref.read(transferProvider.notifier);
        final result = await transferViewModel.executeTransfer();
        
        if (result) {
          final state = ref.read(transferProvider);
          // 송금 완료 페이지로 이동
          context.push(
            FinanceRoutes.getTransferCompletePath(),
            extra: {
              'withdrawalAccountNo': state.withdrawalAccountNo,
              'depositAccountNo': state.depositAccountNo,
              'amount': state.amount,
            }
          );
        } else {
          // 송금 실패 처리
          final transferState = ref.read(transferProvider);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('송금에 실패했습니다: ${transferState.error ?? "알 수 없는 오류"}')),
          );
          context.pop(); // 인증 페이지 닫기
        }
      }
    } else {
      // 인증 실패
      setState(() {
        _isLoading = false;
        _wrongAttempts++;
        _pinValue = '';
        _pinController.text = '';
      });

      if (_wrongAttempts >= 5) {
        // 5회 실패 시 뒤로 가기
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('인증 시도 횟수를 초과했습니다.')),
          );
          context.pop();
        }
      } else {
        // 오류 메시지 표시
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('인증번호가 일치하지 않습니다. (${_wrongAttempts}/5)')),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppbar(
        title: 'my little 자산',
        backgroundColor: AppColors.background,
      ),
      body: _isLoading 
        ? const Center(child: CircularProgressIndicator())
        : Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      '인증서 비밀번호',
                      style: AppTextStyles.appBar,
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      '비밀번호 6자리를 입력해주세요',
                      style: TextStyle(fontSize: 14, color: AppColors.blackLight),
                    ),
                    const SizedBox(height: 40),
                    PinDotsRow(
                      // 현재 입력된 숫자 위치를 공유해야 색이 채워져요!
                      currentDigit: _pinValue.length,
                      // 탭했을 때 키보드 올라오는 함수를 공유하세요
                      onTap: _showKeyboard,
                    ),
                    
                  ],
                ),
              ],
            ),
          ),
    );
  }
}