import 'package:go_router/go_router.dart';
import 'package:intl/date_symbols.dart';
import 'package:flutter/material.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/di/providers/auth/identity_verification_provider.dart';
import 'package:marshmellow/di/providers/auth/user_provider.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/utils/lifecycle/app_lifecycle_manager.dart'; 
import 'package:marshmellow/core/config/app_config.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/modal/modal.dart';
import 'package:marshmellow/presentation/widgets/text_input/text_input.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/keyboard/index.dart';
import 'package:marshmellow/presentation/pages/auth/widgets/etc/custom_button.dart';

// 본인인증 단계 관리 Provider
final verificationStepProvider = StateProvider<int>((ref) => 0);

// 본인인증 입력값 관리 Provider
final nameProvider = StateProvider<String>((ref) => '');
final idNumProvider = StateProvider<String>((ref) => '');
final phoneProvider = StateProvider<String>((ref) => '');
final carrierProvider = StateProvider<String?>((ref) => null);


class SignupPage extends ConsumerWidget {
  const SignupPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // 라이프사이클 상태 구독
    final lifecycleState = ref.watch(lifecycleStateProvider);
    // 현재 본인인증 단계
    final currentStep = ref.watch(verificationStepProvider);


    return Scaffold(

      body: Padding(
        padding: const EdgeInsets.all(40.0),
        child: Column(
          children: [
            const SizedBox(height: 40),
            const Text('본인인증', style: AppTextStyles.mainTitle),
            const SizedBox(height: 40),

            // 본인인증 단계에 따른 입력 폼
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  children: [
                    // 이름 입력 (항상 표시)
                    NameInputSection(isVisible: currentStep >=0),
                    // 주민등록번호 입력 (본인인증 단계 1 이상일 때 표시)
                    IdNumInputSection(isVisible: currentStep >= 1),
                    // 휴대폰 번호 입력 (본인인증 단계 2 이상일 때 표시)
                    PhoneInputSection(isVisible: currentStep >= 2),
                    // 통신사 선택 (본인인증 단계 3 이상일 때 표시)
                    CarrierSelectSection(isVisible: currentStep >= 3),
                  ],
                ),
              ),
            ),

            // 하단 버튼 영역
            // 본인인증 단계 4 전까지만 '다음' 버튼
            if (currentStep < 4) 
              NextButton()
            // 이후는 '본인인증' 버튼
            else
              VerifyButton(onPressed: () => _showTermModal(context, ref)),
          ],
        )
      )
    );
  }

  // 약관 모달 표시
  void _showTermModal(BuildContext context, WidgetRef ref) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (context) => Padding(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(context).viewInsets.bottom,
        ),
        child: Modal(
          backgroundColor: AppColors.modalBackground, 
          title: '이용약관 및 개인정보 처리방침',
          titleStyle: AppTextStyles.bodyMedium,
          showDivider: false,
          maxHeight: MediaQuery.of(context).size.height * 0.7,
          child: TermsAgreementModal(),
          ),
      ),
    );
  }
}

// 이름 입력 섹션
class NameInputSection extends ConsumerWidget {
  final bool isVisible;
  final TextEditingController _nameController = TextEditingController();

  NameInputSection({required this.isVisible, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (!isVisible) return const SizedBox.shrink();
    
    final name = ref.watch(nameProvider);

    if (_nameController.text != name) {
      _nameController.text = name;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextInput(
          label: '이름', 
          controller: _nameController, 
          onChanged: (value) => ref.read(nameProvider.notifier).state = value,
        ),
        const SizedBox(height: 8),
      ],
    );
  }
}

// 주민등록번호 입력 섹션
class IdNumInputSection extends ConsumerWidget {
  final bool isVisible;
  final TextEditingController _idNumController = TextEditingController();

  IdNumInputSection({required this.isVisible, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (!isVisible) return const SizedBox.shrink();

    final idNum = ref.watch(idNumProvider);

    // 주민번호 포맷팅
    String formattedIdNum = '';
    if (idNum.isNotEmpty) {
      if (idNum.length <= 6) {
        formattedIdNum = idNum;
      } else {
        formattedIdNum = '${idNum.substring(0, 6)}-${idNum.substring(6)}******';
      }
    }


    _idNumController.text = formattedIdNum;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextInput(
          label: '주민등록번호 앞 7자리', 
          controller: _idNumController, 
          readOnly: true,
          onTap: () {
            // 보안 키패드
            KeyboardModal.showSecureNumericKeyboard(
              context: context, 
              onValueChanged: (value) {
                ref.read(idNumProvider.notifier).state = value;
                _idNumController.text = value;
              },
              initialValue: idNum,
              maxLength: 7,
            );
          },
        ),
        const SizedBox(height: 8),
      ],
    );
  }
}

// 휴대폰 번호 입력 섹션
class PhoneInputSection extends ConsumerWidget {
  final bool isVisible;
  final TextEditingController _phoneController = TextEditingController();

  PhoneInputSection({required this.isVisible, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (!isVisible) return const SizedBox.shrink();

    final phone = ref.watch(phoneProvider);

    // 휴대폰 번호 포맷팅
    String formattedPhone = '';
    if (phone.isNotEmpty) {
      if (phone.length <= 3) {
        formattedPhone = phone;
      } else if (phone.length <= 7) {
        formattedPhone = '${phone.substring(0, 3)}-${phone.substring(3)}';
      } else {
        formattedPhone = '${phone.substring(0, 3)}-${phone.substring(3, 7)}-${phone.substring(7)}';
      }
    }

    _phoneController.text = formattedPhone;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextInput(
          label: '휴대폰 번호', 
          controller: _phoneController, 
          readOnly: true, // 시스템 키보드 사용 방지
          onTap: () {
            // 숫자 키패드
            KeyboardModal.showNumericKeyboard(
              context: context, 
              onValueChanged: (value) {
                ref.read(phoneProvider.notifier).state = value;
                _phoneController.text = value;
              },
              initialValue: phone,
            );
          },
        ),
        const SizedBox(height: 8),
      ],
    );
  }
}

// 통신사 선택 섹션
class CarrierSelectSection extends ConsumerWidget {
  final bool isVisible;
  final TextEditingController _carrierController = TextEditingController();

  CarrierSelectSection({required this.isVisible, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    if (!isVisible) return const SizedBox.shrink();

    final carrier = ref.watch(carrierProvider);
    _carrierController.text = carrier ?? '';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextInput(
          label: '통신사',
          controller: _carrierController,
          readOnly: true,
          onTap: () => _showCarrierSelectModal(context, ref),
        ),
        const SizedBox(height: 8),
      ],
    );
  }
  void _showCarrierSelectModal(BuildContext context, WidgetRef ref) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (context) => Padding(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(context).viewInsets.bottom,
        ),
        child: Modal(
          backgroundColor: AppColors.modalBackground,
          title: '통신사 선택',
          titleStyle: AppTextStyles.bodyMedium,
          showDivider: false,
          child: CarrierSelectModal(
            onSelect: (carrier) {
              ref.read(carrierProvider.notifier).state = carrier;
              _carrierController.text = carrier;
              Navigator.pop(context);
            },
            selectedCarrier: ref.watch(carrierProvider),
          ),
        ),
      )
    );
  }
}

class CarrierSelectModal extends StatelessWidget {
  final Function(String) onSelect;
  final String? selectedCarrier;

  const CarrierSelectModal({
    required this.onSelect,
    this.selectedCarrier,
    Key? key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final carriers = [
      {'name' : 'SKT', 'logoPath': ''},
      {'name' : 'KT', 'logoPath': ''},
      {'name' : 'LG U+', 'logoPath': ''},
      {'name' : 'SKT 알뜰폰', 'logoPath': ''},
      {'name' : 'KT 알뜰폰', 'logoPath': ''},
      {'name' : 'LG U+ 알뜰폰', 'logoPath': ''},
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        const SizedBox(height: 10),
        ...carriers.map((carrier) => _buildCarrierItem(
          context,
          carrier['name']!,
          carrier['logoPath']!,
        )).toList(),
        const SizedBox(height: 20),
      ],
    );
  }

  Widget _buildCarrierItem(BuildContext context, String name, String logoPath) {
    final isSelected = selectedCarrier == name;

    return InkWell(
      onTap: () => onSelect(name),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            Icon(
              isSelected ? Icons.check : Icons.check_sharp,
              color: isSelected ? AppColors.blueDark : AppColors.textSecondary.withOpacity(0.3),
              size: 20,
            ),
            const SizedBox(width: 20),
            Expanded(
              child: Text(
                name,
                style: AppTextStyles.bodyLargeLight.copyWith(
                  fontWeight: FontWeight.w300,
                  color: isSelected ? AppColors.textPrimary : AppColors.textSecondary,
                ),
              ),
            ),
          ],
        ),
      )
    );
  }
}

// 다음 버튼
class NextButton extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final currentStep = ref.watch(verificationStepProvider);
    final name = ref.watch(nameProvider);
    final idNum = ref.watch(idNumProvider);
    final phone = ref.watch(phoneProvider);
    final carrier = ref.watch(carrierProvider);

    // 현재 단계별 버튼 활성화 조건
    bool isButtonEnabled = false;

    switch (currentStep) {
      case 0:
        isButtonEnabled = name.isNotEmpty;
        break;
      case 1:
        isButtonEnabled = idNum.length == 7;
        break;
      case 2:
        isButtonEnabled = phone.length >= 10;
        break;
      case 3:
        isButtonEnabled = carrier != null;
        break;
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: CustomButton(
        text:'다음',
        onPressed: isButtonEnabled ? () {
          ref.read(verificationStepProvider.notifier).state++;
        } : null,
        isEnabled:isButtonEnabled,
      ),
    );
  }
}

// 본인인증 버튼
class VerifyButton extends ConsumerWidget {
  final VoidCallback onPressed;

  const VerifyButton({required this.onPressed, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final name = ref.watch(nameProvider);
    final idNum = ref.watch(idNumProvider);
    final phone = ref.watch(phoneProvider);
    final carrier = ref.watch(carrierProvider);

    final isButtonEnabled = name.isNotEmpty && idNum.length == 7 && phone.length >= 10 && carrier != null;

    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: CustomButton(
        text: '본인인증하기',
        onPressed: onPressed,
        isEnabled: true,
      ),
    );
  }
}


// 약관 동의 모달
class TermsAgreementModal extends ConsumerStatefulWidget {
  @override
  _TermsAgreementModalState createState() => _TermsAgreementModalState();
}

class _TermsAgreementModalState extends ConsumerState<TermsAgreementModal> {
  bool agreeAll = false;
  bool agreeTerms = false;
  bool agreePrivacy = false;
  bool agreeMMbank = false;
  bool agreeMMstock = false;
  
  @override
  Widget build(BuildContext context) {
    // 인증 상태 관찰
    final verificationState = ref.watch(identityVerificationProvider);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        const SizedBox(height: 10),
        // 전체 동의
        _buildAgreementCheckbox(
          title: '전체 동의',
          isChecked: agreeAll,
          onChanged: (value) {
            setState(() {
              agreeAll = value ?? false;
              agreeTerms = agreeAll;
              agreePrivacy = agreeAll;
              agreeMMbank = agreeAll;
              agreeMMstock = agreeAll;
            });
          },
          isAllAgree: true,
        ),

        const SizedBox(height: 0),
        
        // 이용 약관 동의
        _buildAgreementCheckbox(
          title: '필수 MM 이용 약관 및 동의사항',
          isChecked: agreeTerms,
          onChanged: (value) {
            setState(() {
              agreeTerms = value ?? false;
              _updateAllAgreeStatus();
            });
          },
          showDetail: true,
        ),
        
        // 개인정보 수집 및 이용 동의
        _buildAgreementCheckbox(
          title: '필수 본인 확인 약관 및 동의사항',
          isChecked: agreePrivacy,
          onChanged: (value) {
            setState(() {
              agreePrivacy = value ?? false;
              _updateAllAgreeStatus();
            });
          },
          showDetail: true,
        ),
        
        // mm뱅크 서비스 제공 동의
        _buildAgreementCheckbox(
          title: '선택 MM뱅크 서비스 제공 동의',
          isChecked: agreeMMbank,
          onChanged: (value) {
            setState(() {
              agreeMMbank = value ?? false;
              _updateAllAgreeStatus();
            });
          },
          showDetail: true,
        ),

        // mm증권 서비스 제공 동의
        _buildAgreementCheckbox(
          title: '선택 MM증권 서비스 제공 동의',
          isChecked: agreeMMstock,
          onChanged: (value) {
            setState(() {
              agreeMMstock = value ?? false;
              _updateAllAgreeStatus();
            });
          },
          showDetail: true,
        ),
        
        const SizedBox(height: 10),

        // 인증 요청 상태(본인확인및SSE연결까지)에 따른 표시
        if (verificationState.status == VerificationStatus.loading)
          const Center(child: CircularProgressIndicator())
        else if (verificationState.status == VerificationStatus.failed)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8.0),
            child: Text(
              verificationState.errorMessage ?? '인증에 실패했습니다.',
              style: AppTextStyles.bodySmall.copyWith(color: Colors.red),
              textAlign: TextAlign.center,
            ),
          ),
        
        // 본인인증하기 버튼
        CustomButton(
          text: '동의하고 본인 인증하기',
          onPressed: (agreeTerms && agreePrivacy) 
              ? () => _processVerification(context)
              : null,
          isEnabled: agreeTerms && agreePrivacy,
        ),
        
        // 동의 안함 버튼
        Center(
          child: TextButton(
            onPressed: () {
              Navigator.pop(context);
            },
            child: Text(
              '동의 안함',
              style: AppTextStyles.bodySmall.copyWith(
                color: AppColors.textSecondary,
              ),
            ),
          ),
        ),
      ],
    );
  }

  // 본인확인 처리 메서드
  void _processVerification(BuildContext context) async {
    final phone = ref.read(phoneProvider);
    final name = ref.read(nameProvider);
    final idNum = ref.read(idNumProvider);
    final carrier = ref.read(carrierProvider) ?? '';

    // 본인확인 요청
    await ref.read(identityVerificationProvider.notifier).verifyIdentity(phone);

    // 확인 상태 확인
    final state = ref.read(identityVerificationProvider);

    if (state.status == VerificationStatus.emailSent) {
      // 사용자 정보 저장
      await ref.read(userStateProvider.notifier).setVerificationData(
        userName: name, 
        phoneNumber: phone, 
        userCode: idNum, 
        carrier: carrier
      );
      
      // 인증 메시지 페이지로 이동
      // 인증코드 발송 성공 - 모달 닫기 및 메시지 페이지로 이동
      Navigator.pop(context);
      context.go(SignupRoutes.getAuthMessagePath(), extra: {
        'name': ref.read(nameProvider),
        'idNum': ref.read(idNumProvider),
        'phone': ref.read(phoneProvider),
        'carrier': ref.read(carrierProvider),
        'serverEmail': state.serverEmail,
        'verificationCode': state.verificationCode,
        'expiresIn': state.expiresIn,
      });
    }
  }
  
  Widget _buildAgreementCheckbox({
    required String title,
    required bool isChecked,
    required ValueChanged<bool?> onChanged,
    bool showDetail = false,
    bool isAllAgree = false,
  }) {
    return InkWell(
      onTap: () => onChanged(!isChecked),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            Icon(
              isChecked ? Icons.check: Icons.check_sharp,
              color: isChecked ? AppColors.blueDark : AppColors.textSecondary.withOpacity(0.3),
              size: 20,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                title,
                style: AppTextStyles.bodyMediumLight.copyWith(
                  fontWeight: FontWeight.w300,
                  color: isChecked ? AppColors.textPrimary : AppColors.textSecondary,
                ),
              ),
            ),
            if (showDetail)
              IconButton(
                icon: const Icon(Icons.arrow_forward_ios, size: 10),
                onPressed: () {
                  // 약관 상세 페이지로 이동
                },),
          ],
        ),
        ),
    );
  }
  
  void _updateAllAgreeStatus() {
    // 필수 항목들과 선택 항목이 모두 체크되었는지 확인
    setState(() {
      agreeAll = agreeTerms && agreePrivacy && agreeMMbank && agreeMMstock;
    });
  }
}