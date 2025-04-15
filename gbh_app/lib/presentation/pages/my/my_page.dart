import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/config/app_config.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/certification_select_content.dart';
import 'package:marshmellow/presentation/viewmodels/encryption/encryption_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_secure_info_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_info_viewmodel.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/services.dart';

class MyPage extends ConsumerStatefulWidget {
  const MyPage({super.key});

  @override
  ConsumerState<MyPage> createState() => _MyPageState();
}

class _MyPageState extends ConsumerState<MyPage> {
  // 월급 수정 모드 플래그
  bool _isEditingSalary = false;
  
  // 수정 모드용 컨트롤러들
  late TextEditingController _salaryController;
  late TextEditingController _dateController;
  late TextEditingController _accountController;
  
  // 에러 상태
  String? _salaryError;
  String? _dateError;
  String? _accountError;
  
  @override
  void initState() {
    super.initState();
    _salaryController = TextEditingController();
    _dateController = TextEditingController();
    _accountController = TextEditingController();
  }
  
  @override
  void dispose() {
    _salaryController.dispose();
    _dateController.dispose();
    _accountController.dispose();
    super.dispose();
  }
  
  // 수정 모드 시작
  void _startEditingSalary(int? salary, int? date, String? account) {
    _salaryController.text = salary?.toString() ?? '';
    _dateController.text = date?.toString() ?? '';
    _accountController.text = account ?? '';
    
    setState(() {
      _isEditingSalary = true;
      _salaryError = null;
      _dateError = null;
      _accountError = null;
    });
  }
  
  // 수정 모드 취소
  void _cancelEditingSalary() {
    setState(() {
      _isEditingSalary = false;
    });
  }
  
  // 필드 유효성 검사
  bool _validateFields() {
    bool isValid = true;
    
    // 급여 검증
    if (_salaryController.text.isEmpty) {
      setState(() => _salaryError = '월급을 입력해주세요');
      isValid = false;
    } else if (int.tryParse(_salaryController.text) == null) {
      setState(() => _salaryError = '숫자만 입력해주세요');
      isValid = false;
    } else {
      setState(() => _salaryError = null);
    }
    
    // 날짜 검증
    if (_dateController.text.isEmpty) {
      setState(() => _dateError = '월급일을 입력해주세요');
      isValid = false;
    } else if (int.tryParse(_dateController.text) == null) {
      setState(() => _dateError = '숫자만 입력해주세요');
      isValid = false;
    } else {
      final date = int.parse(_dateController.text);
      if (date < 1 || date > 31) {
        setState(() => _dateError = '1~31 사이의 날짜를 입력해주세요');
        isValid = false;
      } else {
        setState(() => _dateError = null);
      }
    }
    
    // 계좌 검증
    if (_accountController.text.isEmpty) {
      setState(() => _accountError = '계좌번호를 입력해주세요');
      isValid = false;
    } else {
      setState(() => _accountError = null);
    }
    
    return isValid;
  }
  
  // 월급 정보 저장
  Future<void> _saveSalaryInfo() async {
    if (!_validateFields()) return;
    
    final salary = int.parse(_salaryController.text);
    final date = int.parse(_dateController.text);
    final account = _accountController.text;
    
    final userInfoState = ref.read(userInfoProvider);
    
    try {
      // 기존 정보 여부에 따라 등록 또는 수정
      if (userInfoState.userDetail.salaryAmount != null && 
          userInfoState.userDetail.salaryDate != null &&
          userInfoState.userDetail.salaryAccount != null) {
        await ref.read(userInfoProvider.notifier).myUpdateSalary(salary, date, account);
      } else {
        await ref.read(userInfoProvider.notifier).myRegisterSalary(salary, date, account);
      }
      
      setState(() {
        _isEditingSalary = false;
      });
      
      // 성공 메시지 표시
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('월급 정보가 저장되었습니다'))
      );
    } catch (e) {
      // 에러 메시지 표시
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('오류가 발생했습니다: $e'))
      );
    }
  }
  
Widget _buildInfoButton({
  required String label, 
  required String value, 
  required VoidCallback onPressed,
  VoidCallback? onLongPress,  // 추가: 길게 누르기 콜백
  bool showIcon = false,
  bool isHighlighted = false,
}) {
  return GestureDetector(  // ElevatedButton 대신 GestureDetector 사용
    onTap: onPressed,
    onLongPress: onLongPress,  // 길게 누르기 처리
    child: Container(
      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: isHighlighted ? Colors.grey[100] : Colors.transparent,
        border: Border.all(color: Colors.grey[300]!, width: 1),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: TextStyle(
                  color: Colors.grey[600],
                  fontSize: 12,
                ),
              ),
              SizedBox(height: 4),
              Text(
                value,
                style: AppTextStyles.bodyMedium.copyWith(color: AppColors.blackPrimary)
              ),
            ],
          ),
          if (showIcon) SvgPicture.asset(IconPath.caretRight),
        ],
      ),
    ),
  );
}
  
// 편집 모드용 필드 위젯
Widget _buildEditField({
  required String label,
  required TextEditingController controller,
  String? errorText,
  String? hintText,
  TextInputType keyboardType = TextInputType.text,
}) {
  return Container(
    padding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
    decoration: BoxDecoration(
      border: Border.all(color: Colors.grey[300]!, width: 1),
      borderRadius: BorderRadius.circular(8),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.grey[600],
            fontSize: 12,
          ),
        ),
        SizedBox(height: 4),
        TextField(
          controller: controller,
          keyboardType: keyboardType,
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 14,
          ),
          decoration: InputDecoration(
            hintText: hintText,
            errorText: errorText,
            isDense: true,
            contentPadding: EdgeInsets.symmetric(horizontal: 0, vertical: 0),
            border: InputBorder.none,
            enabledBorder: InputBorder.none,
            focusedBorder: InputBorder.none,
          ),
          onChanged: (_) => _validateFields(),
        ),
        if (errorText != null)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              errorText,
              style: TextStyle(
                color: AppColors.warnningLight,
                fontSize: 12,
              ),
            ),
          ),
      ],
    ),
  );
}
  
  String _formatCurrency(int? amount) {
    if (amount == null) return '정보 없음';
    return '${amount.toString().replaceAllMapped(RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'), (Match m) => '${m[1]},')}원';
  }
  
  @override
  Widget build(BuildContext context) {
    final userInfoState = ref.watch(userInfoProvider);
    final userSecureInfoState = ref.watch(userSecureInfoProvider);
    
    // 월급 정보 상태 확인
    final hasSalaryInfo = userInfoState.userDetail.salaryAmount != null && 
                          userInfoState.userDetail.salaryDate != null &&
                          userInfoState.userDetail.salaryAccount != null;
    
    return Scaffold(
      appBar: CustomAppbar(
        title: '마이구미 🍇',
        actions: [
          if (AppConfig.isDevelopment())
            IconButton(
              icon: const Icon(Icons.bug_report),
              onPressed: () {
                context.push(FinanceRoutes.getTestPath());
              },
              tooltip: '테스트 페이지로 이동',
            ),
          IconButton(
              icon: const Icon(Icons.refresh),
              color: AppColors.blackPrimary,
              onPressed: () {
                ref.read(userInfoProvider.notifier).loadAllUserInfo();
              },
            ),
        ],
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: userInfoState.isLoading
            ? Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Image.asset(
                'assets/images/characters/char_angry_notebook.png', 
                width: 180,
                height: 180,
              ),
              const SizedBox(height: 16),
              const Text(
                '앗! 마이 데이터를 불러올 수 없어요',
                style: AppTextStyles.bodyMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
                      Button(
                        onPressed: () => ref.read(userInfoProvider.notifier).loadAllUserInfo(),
                        text: '보안 인증',
                      ),
            ],
          ),
        )
                : SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        SizedBox(height: 8),
                        Padding(
                          padding: const EdgeInsets.fromLTRB(8.0, 0, 0, 0),
                          child: Text(
                            '내 정보',
                            style: AppTextStyles.appBar.copyWith(color: AppColors.blackPrimary)
                          ),
                        ),
                        SizedBox(height: 16),
                        _buildInfoButton(
                          label: '이름',
                          value: userSecureInfoState.userName ?? '정보 없음',
                          onPressed: () {
                            // 이름 관련 동작
                          },
                        ),
                        SizedBox(height: 12),
                        _buildInfoButton(
                          label: '전화번호',
                          value: userSecureInfoState.phoneNumber ?? '정보 없음',
                          onPressed: () {
                            // 전화번호 관련 동작
                          },
                        ),
                        SizedBox(height: 12),
                        _buildInfoButton(
                          label: '이메일',
                          value: userSecureInfoState.certificateEmail ?? '정보 없음',
                          onPressed: () {
                            // 이메일 관련 동작
                          },
                        ),
                        SizedBox(height: 24),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Padding(
                              padding: const EdgeInsets.fromLTRB(8.0, 0, 0, 0),
                              child: Text(
                                '월급 정보',
                                style: AppTextStyles.appBar.copyWith(color: AppColors.blackPrimary)
                              ),
                            ),
                            // 편집 모드가 아닐 때만 편집 버튼 표시
                            if (!_isEditingSalary)
                            Row(
                              children: [
                                InkWell(
                                  onTap: () {
                                    Clipboard.setData(ClipboardData(text: userInfoState.userDetail.salaryAccount ?? '정보 없음'));
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      const SnackBar(content: Text('계좌번호가 복사되었습니다')),
                                    );
                                  },
                                  borderRadius: BorderRadius.circular(4),
                                  child: Padding(
                                    padding: const EdgeInsets.all(4),
                                    child: SvgPicture.asset(
                                      'assets/icons/body/CopySimple.svg',
                                      height: 20,
                                      color: AppColors.blackLight,
                                    ),
                                  ),
                                ),
                                SizedBox(width: 10,),
                                GestureDetector(
                                  onTap: () => _startEditingSalary(
                                    userInfoState.userDetail.salaryAmount,
                                    userInfoState.userDetail.salaryDate,
                                    userInfoState.userDetail.salaryAccount,
                                  ),
                                  child: Container(
                                    padding: EdgeInsets.fromLTRB(10.0, 5.0, 10.0, 5.0),
                                    // decoration: BoxDecoration(
                                    //   color: AppColors.backgroundBlack,
                                    //   borderRadius: BorderRadius.circular(5.0),
                                    // ),
                                    child: SvgPicture.asset(IconPath.pencilSimple, color: AppColors.backgroundBlack,)
                                  ),
                                ) 
                              ],
                            )else
                            Row(
                              children: [
                                SizedBox(width: 12),
                                Button(
                                  width: 45,
                                  height: 30,
                                  color: AppColors.blackLight,
                                  textStyle: AppTextStyles.bodySmall.copyWith(color: AppColors.background),
                                  onPressed: _cancelEditingSalary,
                                  text: '취소',
                                ),
                                SizedBox(width: 6),
                                Button(
                                  width: 45,
                                  height: 30,
                                  color: AppColors.blackLight,
                                  textStyle: AppTextStyles.bodySmall.copyWith(color: AppColors.background),
                                  onPressed: _saveSalaryInfo,
                                  text: '저장',
                                ),
                              ],
                            ),
                          ],
                        ),
                        SizedBox(height: 16),
                        // 월급 정보 섹션 - 편집 모드일 때와 아닐 때 다른 UI 표시
                        if (_isEditingSalary) 
                          // 편집 모드 UI
                          Card(
                            color: AppColors.background,
                            elevation: 0,
                            // margin: EdgeInsets.only(bottom: 16),
                            child: Container(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  _buildEditField(
                                    label: '급여 계좌번호',
                                    controller: _accountController,
                                    errorText: _accountError,
                                    hintText: '급여를 받는 계좌번호',
                                  ),
                                  SizedBox(height: 12),
                                  _buildEditField(
                                    label: '월급액 (원)',
                                    controller: _salaryController,
                                    errorText: _salaryError,
                                    hintText: '숫자만 입력 (예: 3000000)',
                                    keyboardType: TextInputType.number,
                                  ),
                                  SizedBox(height: 12),
                                  _buildEditField(
                                    label: '월급일 (1~31)',
                                    controller: _dateController,
                                    errorText: _dateError,
                                    hintText: '매월 급여일 (예: 15)',
                                    keyboardType: TextInputType.number,
                                  ),
                                  SizedBox(height: 16),
                                ],
                              ),
                            ),
                          )
                        else 
                          // 일반 모드 UI
                          Column(
                            children: [
                              _buildInfoButton(
                                label: '월급 계좌',
                                value: userInfoState.userDetail.salaryAccount ?? '정보 없음',
                                onPressed: () {
                                  _startEditingSalary(
                                    userInfoState.userDetail.salaryAmount,
                                    userInfoState.userDetail.salaryDate,
                                    userInfoState.userDetail.salaryAccount,
                                  );
                                },
                              ),
                              SizedBox(height: 12),
                              _buildInfoButton(
                                label: '월급액',
                                value: _formatCurrency(userInfoState.userDetail.salaryAmount),
                                onPressed: () {
                                  _startEditingSalary(
                                    userInfoState.userDetail.salaryAmount,
                                    userInfoState.userDetail.salaryDate,
                                    userInfoState.userDetail.salaryAccount,
                                  );
                                },
                              ),
                              SizedBox(height: 12),
                              _buildInfoButton(
                                label: '급여일',
                                value: userInfoState.userDetail.salaryDate != null 
                                      ? '매월 ${userInfoState.userDetail.salaryDate}일'
                                      : '정보 없음',
                                onPressed: () {
                                  _startEditingSalary(
                                    userInfoState.userDetail.salaryAmount,
                                    userInfoState.userDetail.salaryDate,
                                    userInfoState.userDetail.salaryAccount,
                                  );
                                },
                              ),
                            ],
                          ),
                        SizedBox(height: 24),
                        Padding(
                          padding: const EdgeInsets.fromLTRB(8.0, 0, 0, 0),
                          child: Text(
                            '인증 정보',
                            style: AppTextStyles.appBar.copyWith(color: AppColors.blackPrimary)
                          ),
                        ),
                        SizedBox(height: 16),
_buildInfoButton(
  label: '인증서',
  value: '내 금융인증서 관리',
  onPressed: () {
    // 기존 코드 유지
    ref.read(aesKeyNotifierProvider.notifier).fetchAesKey();
    
    showCertificateModal(
      context: context, 
      ref: ref, 
      userName: userSecureInfoState.userName ?? '사용자', 
      title: '금융인증서 관리',
      expiryDate: '2028.03.14.', 
      onConfirm: () {
        // 인증서 확인 후 처리할 로직
      }
    );
  },
  onLongPress: () async {
    // FCM 토큰 새로 요청 및 출력
    String? token = await FirebaseMessaging.instance.getToken();
    
    if (token != null) {
      // 콘솔에 출력 (디버그 모드에서만 확인 가능)
      print("📱 FCM Token refreshed: $token");
      
      // 릴리스 모드에서도 확인할 수 있도록 팝업 표시
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text('FCM 토큰 새로고침'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('새로 발급된 FCM 토큰:'),
              SizedBox(height: 8),
              Container(
                padding: EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(4),
                ),
                child: SelectableText(
                  token,
                  style: TextStyle(fontSize: 12),
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () {
                Clipboard.setData(ClipboardData(text: token));
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('토큰이 클립보드에 복사되었습니다')),
                );
              },
              child: Text('복사'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text('닫기'),
            ),
          ],
        ),
      );

      // 원하는 경우: 토큰을 서버로 직접 전송하는 로직 추가
      // await sendTokenToServer(token);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('FCM 토큰을 가져올 수 없습니다')),
      );
    }
  },
  showIcon: true,
  isHighlighted: true,
),
                        SizedBox(height: 32),
                        // Button(
                        //   onPressed: () => ref.read(userInfoProvider.notifier).loadAllUserInfo(),
                        //   text: '새로고침',
                        // ),
                        // SizedBox(height: 16), // 스크롤 시 여백 확보
                      ],
                    ),
                  ),
      ),
    );
  }
}