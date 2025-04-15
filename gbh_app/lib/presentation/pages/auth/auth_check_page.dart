import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:marshmellow/core/services/user_preferences_service.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/di/providers/auth/pin_provider.dart';
import 'package:marshmellow/di/providers/auth/user_provider.dart';
import 'package:marshmellow/di/providers/core_providers.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_info_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';

/*
  앱 시작시 로그인 회원가입 여부 확인 페이지
*/
class AuthCheckPage extends ConsumerStatefulWidget {
  const AuthCheckPage({Key? key}) : super(key: key);

  @override
  ConsumerState<AuthCheckPage> createState() => _AuthCheckPageState();
}

class _AuthCheckPageState extends ConsumerState<AuthCheckPage> {
  @override
  void initState() {
    super.initState();
    print('===== AuthCheckPage initState 실행 =====');
    _checkAuthStatus();
  }

  // 인증 상태 확인
  Future<void> _checkAuthStatus() async {
    

    // 개발용 자동 로그인 코드 (출시 전 제거)
    // TODO: 출시 전 이 부분 삭제
    final secureStorage = ref.read(secureStorageProvider);
    await secureStorage.write(
        key: StorageKeys.phoneNumber, value: '01056297169');
    await secureStorage.write(key: StorageKeys.userName, value: '윤잰큰');
    

    // <<<<<<<<<<<< [ 어세스 토큰을 이 아래에 넣으세요 ] <<<<<<<<<<<<<<<<<<<<<<<<
    await secureStorage.write(key: StorageKeys.accessToken, value: 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJ0b2tlblR5cGUiOiJBQ0NFU1MiLCJ1c2VyUGsiOjMsInN1YiI6ImFjY2Vzcy10b2tlbiIsImlhdCI6MTc0NDI5NTU0NSwiZXhwIjoxNzQ2MDk1NTQ1fQ.gXwkqoQjoGypnx2HM3DDBTs1U5oIYO5xLKQX_f3sbonvzY5uR5j-whzaI5FwmE5iTybffAWgFWK6C0MnKhhC-w'); 
    await secureStorage.write(key: StorageKeys.refreshToken, value: 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJ0b2tlblR5cGUiOiJSRUZSRVNIIiwidXNlclBrIjozLCJzdWIiOiJyZWZyZXNoLXRva2VuIiwiaWF0IjoxNzQ0Mjk1NTQ1LCJleHAiOjE3NzAyMTU1NDV9.AIH90MmA-pFiAdRM-KbG0w4d75PXqLu5jSP_itA7IZakXdB_kZCwxqQe6tAMFiLQWFyyfknewt4V8omvYyDo9w');
    await secureStorage.write(key: StorageKeys.certificatePem, value: '-----BEGIN CERTIFICATE-----MIIC5DCCAcygAwIBAgIGAZYQxKmeMA0GCSqGSIb3DQEBDQUAMCwxDjAMBgNVBAMMBU1NIENBMQ0wCwYDVQQKDARNeUNBMQswCQYDVQQGEwJLUjAeFw0yNTA0MDcxNTAwNDZaFw0yNjA0MDcxNTAwNDZaMDoxCzAJBgNVBAYTAktSMQwwCgYDVQQKEwNHQkgxHTAbBgNVBAMTFGdpbmllZTE3NTBAbmF2ZXIuY29tMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAufEYp+EbHTLEW24swo/4/zuNNLz+nczUcLGnNImlCOgbza0Tt3VpDW0aNLRLm8K742UxCWXH3oLSOiyjVifyzlKNfsC2+4fJ8QDvONaXImhFCV9teckb+zwhypYbMlDcpFDNiVf1++nEqzmLZzZ1j8r9xmxeRNfkdt8hikbaLaPGIKcWrC7HKeBPvUijZhx5J5WZajGIUjajz46Gz6sPN6cq28DY4TxdZgQRTSlALnUGlG6oyX8WqFdwJf5WsdZ5l5GmotsouPmcIQZ8BswELLIYKes1LZ11fHEgLl2tW5PF8xL+3gMzyJ5IFV/BHuyKQx3HRAqNDNlobt66h3z5BQIDAQABMA0GCSqGSIb3DQEBDQUAA4IBAQAuZfE4JiTeN/ML51WwgHvQ3TwrR8bFVHZp3TWbjWh6jTUsv+4o5i751g8UONFYNNhe8mCNECyjXeAi1R75+iUGE9I6NTovg6vugvFo0rqukX8Nx2t2n/af2M1YETPxy26UfSG8quwTUgWn/RSRHusYQ0CxKx7MQ7kS1RR14uIastrcZUyGr/Od+zA9MClETQ/xTDWkIr4CZp8w1pcrJKGnW7eWYPPL2UOMGmJ6KBszZ3q7fWf59rfU2qRqM+YDrgSJKjyrzEXJ1c//OIS6eT+8k+soN6C6xPddj4qqRy+pW6Ff7Ngl2/271/aMb2KJfmBZz9eCgMtUy5QpSan39rYy-----END CERTIFICATE-----'); 
    await secureStorage.write(key: StorageKeys.userkey, value: '2c2fd595-4118-4b6c-9fd7-fc811910bb75');
                                                                                                                                                                                                                                            
    // secure storage에서 필요한 정보 불러오기
    
    // final secureStorage = ref.read(secureStorageProvider);
    final phoneNumber = await secureStorage.read(key: StorageKeys.phoneNumber);
    final accessToken = await secureStorage.read(key: StorageKeys.accessToken);
    final refreshToken =
        await secureStorage.read(key: StorageKeys.refreshToken);

    print('디버그 - 저장된 정보:');
    print('phoneNumber: $phoneNumber');
    print('accessToken: ${accessToken != null ? '있음' : '없음'}');
    print('refreshToken: ${refreshToken != null ? '있음' : '없음'}');

    // 지연 효과
    await Future.delayed(const Duration(milliseconds: 800));

    if (mounted) {
      if (phoneNumber != null && accessToken != null && refreshToken != null) {
        print('케이스 1: 전화번호와 토큰 모두 있음');
        // 전화번호와 토큰 저장되 있으면 사용자 정보 갱신
        final userNotifier = ref.read(userStateProvider.notifier);

        // 기존 정보 가져와서 함께 설정
        final userName =
            await secureStorage.read(key: StorageKeys.userName) ?? '';
        final userCode =
            await secureStorage.read(key: StorageKeys.userCode) ?? '';
        final carrier =
            await secureStorage.read(key: StorageKeys.carrier) ?? '';

        print('사용자 정보 설정:');
        print('userName: $userName');
        print('userCode: $userCode');
        print('carrier: $carrier');

        await userNotifier.setVerificationData(
            userName: userName,
            phoneNumber: phoneNumber,
            userCode: userCode,
            carrier: carrier);

        print('사용자 정보 설정 완료');

        // 토큰 유효성 검사
        print('토큰 유효성 검사 시작');
        _validateToken();
        // 검사하여 메인 페이지 또는 로그인 페이지로 이동
      } else if (phoneNumber != null) {
        print('케이스 2: 전화번호만 있음');
        // 전화번호만 저장되어 있으면 로그인 페이지로 이동
        final userNotifier = ref.read(userStateProvider.notifier);

        // 기존 정보 가져와서 함께 설정
        final userName =
            await secureStorage.read(key: StorageKeys.userName) ?? '';
        final userCode =
            await secureStorage.read(key: StorageKeys.userCode) ?? '';
        final carrier =
            await secureStorage.read(key: StorageKeys.carrier) ?? '';

        await userNotifier.setVerificationData(
            userName: userName,
            phoneNumber: phoneNumber,
            userCode: userCode,
            carrier: carrier);

        print('PIN 번호 생성 페이지로 이동: ${SignupRoutes.getPinSetupPath()}');
        context.go(SignupRoutes.getPinSetupPath());
      } else {
        print('케이스 3: 정보 없음');
        // 아무 정보도 없으면 회원가입 페이지로 이동
        print('회원가입 페이지로 이동: ${SignupRoutes.root}');
        context.go(SignupRoutes.root);
      }
    }
  }

  Future<void> _validateToken() async {
    final authRepository = ref.read(authRepositoryProvider);
    final secureStorage = ref.read(secureStorageProvider);

    try {
      // 토큰 재발급 시도
      print('토큰 재발급 시도');

      // <<<<<<<<<<<< [ 잰큰 사용할 때 여기도 수정! ] <<<<<<<<<<<<<<<<<<<<<<<<
      // 개발용 자동 로그인 코드 (출시 전 제거)
      // final isValid = await authRepository.reissueToken();
      final isValid = true;
      print('토큰 재발급 결과: $isValid');

      if (mounted) {
        if (isValid) {
          // 토큰이 유효하면 인증서와 userkey 확인
          final certificatePem =
              await secureStorage.read(key: StorageKeys.certificatePem);
          final userkey = await secureStorage.read(key: StorageKeys.userkey);

          print('🪪🪪인증서 확인: ${certificatePem != null ? '있음' : '없음'}');
          print('🪪🪪유저키 확인: ${userkey != null ? '있음' : '없음'}');

          if (certificatePem != null && userkey != null) {
            print('인증서와 유저키 모두 있음: 월급날 체크 시작!');
            
            await _checkSalaryDay();

          } else {
            // 토큰 유효하고
            // 인증서나 유저키가 없으면 인증서 만들러 가기
            // : splash page 에서 한 번 더 조건 필터링 합니다
            print('토큰 유효: 인증서 만들러 가기');
            context.go(SignupRoutes.getMyDataSplashPath());
          }
        } else {
          // 유효하지 않으면 로그인 페이지로 이동
          print('토큰 유효하지 않음: PIN 로그인 페이지로 이동');
          context.go(SignupRoutes.getPinLoginPath());
        }
      }
    } catch (e) {
      print('토큰 재발급 오류 발생: $e');
      if (mounted) {
        // 오류 발생시 로그인 페이지로 이동
        print('오류로 인해 PIN 로그인 페이지로 이동');
        context.go(SignupRoutes.getPinLoginPath());
      }
    }
  }

  // 월급날 확인 메서드 (새로 추가)
  Future<void> _checkSalaryDay() async {
    print('Ⓜ️Ⓜ️월급일 확인 로직 시작');
    
    // 이번 달에 이미 월급날 플로우를 봤는지 확인 (옵션)
    bool hasSeenThisMonth = await UserPreferencesService.hasSeenSalaryFlowThisMonth();
    if (hasSeenThisMonth) {
      print('Ⓜ️Ⓜ️이번 달에 이미 월급날 플로우를 확인함');
      context.go('/budget');
      return;
    }

    try {
      // 사용자 정보 로드
      final userInfoState = ref.read(userInfoProvider);
      
      // 유저 정보가 로드될 때까지 잠시 대기
      if (userInfoState.isLoading) {
        await Future.delayed(const Duration(seconds: 1));
      }
      
      // 사용자의 월급날 확인
      final userDetail = userInfoState.userDetail;
      final salaryDate = userDetail.salaryDate ?? 11; // 기본값 11일
      
      print('Ⓜ️Ⓜ️오늘이 월급날인지 확인하겠습니다. 사용자 월급날 = $salaryDate');
      
      // 오늘이 월급날인지 확인
      final now = DateTime.now();
      if (now.day == salaryDate) {
        print('Ⓜ️Ⓜ️오늘은 월급날입니다!: ${now.day} = $salaryDate');
        
        // 월급날이면 축하 플로우로 이동
        if (mounted) {
          context.go(SignupRoutes.getBudgetCelebratePath());
          
          // 이번 달에 본 것으로 표시 (옵션)
          // await UserPreferencesService.markSalaryFlowSeenThisMonth();
        }
      } else {
        print('Ⓜ️Ⓜ️오늘은 월급날이 아닙니다!: ${now.day} != $salaryDate');
        
        // 월급날이 아니면 일반 예산 페이지로 이동
        if (mounted) {
          context.go('/budget');
        }
      }
    } catch (e) {
      print('Ⓜ️Ⓜ️월급날 확인 중 오류 발생: $e');
      
      // 오류 발생 시 일반 예산 페이지로 이동
      if (mounted) {
        context.go('/budget');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    print('===== AuthCheckPage build 메서드 실행 =====');
    return Scaffold(
        body: CustomLoadingIndicator(
          text: '사회생활의 첫걸음. 재정 관리의 첫걸음.\nMarshMellow',
          backgroundColor: AppColors.whiteLight,
        ));
  }
}
