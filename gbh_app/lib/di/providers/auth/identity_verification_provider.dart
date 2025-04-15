import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/core/config/app_config.dart';
import 'package:marshmellow/data/repositories/auth/identity_verification_repository.dart';
import 'package:marshmellow/di/providers/api_providers.dart';

enum VerificationStatus {
  initial,
  loading,
  emailSent,
  verifying,
  verified,
  expired,
  failed,
  connectionClosed, // sse 연결 종료 감지(임시)
}

// 인증 상태 정보를 담는 클래스
class VerificationState {
  final VerificationStatus status;
  final String? serverEmail;
  final String? verificationCode;
  final int? expiresIn;
  final String? errorMessage;

  VerificationState({
    this.status = VerificationStatus.initial,
    this.serverEmail,
    this.verificationCode,
    this.expiresIn,
    this.errorMessage,
  });

  VerificationState copyWith({
    VerificationStatus? status,
    String? serverEmail,
    String? verificationCode,
    int? expiresIn,
    String? errorMessage,
  }) {
    return VerificationState(
      status: status ?? this.status,
      serverEmail: serverEmail ?? this.serverEmail,
      verificationCode: verificationCode ?? this.verificationCode,
      expiresIn: expiresIn ?? this.expiresIn,
      errorMessage: errorMessage ?? this.errorMessage,
    );
  }
}

// 인증 레포지토리 provider
final authIdentityRepositoryProvider = Provider<AuthIdentityRepository>((ref) {
  final authApi = ref.watch(authApiProvider);
  return AuthIdentityRepository(authApi);
});

// 인증 상태 관리 NotifierProvider
class IdentityVerificationNotifier extends StateNotifier<VerificationState> {
  final AuthIdentityRepository repository;
  StreamSubscription? _sseSubscription;
  http.Client? _httpClient;

  // SSE 파싱을 위한 변수들
  String _currentEvent = '';
  String _currentId = '';
  String _currentData = '';
  bool _isCollectingEvent = false;

  IdentityVerificationNotifier({required this.repository})
    : super(VerificationState());

  // 본인인증 요청
  Future<void> verifyIdentity(String phoneNumber) async {
    try {
      state = state.copyWith(status: VerificationStatus.loading);

      final response = await repository.verifyIdentity(phoneNumber);

      if (response['code'] == 200) {
        final data = response['data'];
        state = state.copyWith(
          status: VerificationStatus.emailSent,
          serverEmail: data['serverEmail'],
          verificationCode: data['code'],
          expiresIn: data['expiresIn'],
        );

        // SSE 연결 설정
        _setupSSEConnection(phoneNumber);
      } else {
        state = state.copyWith(
          status: VerificationStatus.failed,
          errorMessage: response['message'] ?? '본인인증 요청 실패'
        );
      }
    } catch (e) {
      state = state.copyWith(
        status: VerificationStatus.failed,
        errorMessage: '본인인증 요청 중 오류가 발생했습니다: $e',
      );
    }
  }

  DateTime _connectionStartTime = DateTime.now();
  String _lastReceivedData = '';

  // SSE(server-sent events) 연결 설정
  void _setupSSEConnection(String phoneNumber) async {
    try {
      // 기존 연결 종료
      _sseSubscription?.cancel();
      _httpClient?.close();

      // 새 HTTP 클라이언트 생성
      _httpClient = http.Client();

      // sse url 설정
      final sseEndpoint = repository.getIdentityVerificationSubscribeURl(phoneNumber);
      final baseUrl = AppConfig.apiBaseUrl;
      final sseUrl = "$baseUrl$sseEndpoint";


      // http 요청 생성
      final request = http.Request('GET', Uri.parse(sseUrl));
      request.headers.addAll({
        'Accept': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });

      print('SSE URL: $sseUrl');
      print('SSE Headers: ${request.headers}');

      // 요청 전송 및 응답 처리
      final response = await _httpClient!.send(request);

      if (response.statusCode == 200) {
        // 상태 업데이트 - 연결 성공!
        state = state.copyWith(status: VerificationStatus.verifying);

        // 스트림 구독 - 응답 처리
        _sseSubscription = response.stream
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen(
            _processSSELine,
            onError: (error) {
              state = state.copyWith(
                status: VerificationStatus.failed,
                errorMessage: 'SSE 연결 오류: $error',
              );
            },
            onDone: () {
              if (state.status == VerificationStatus.verifying ||
                  state.status == VerificationStatus.emailSent) {
                state = state.copyWith(
                  status: VerificationStatus.connectionClosed,
                  errorMessage: 'SSE 연결이 종료되었습니다.',
                );
              }
            },

          );
      } else {
        state = state.copyWith(
          status: VerificationStatus.failed,
          errorMessage: 'SSE 연결 실패: HTTP ${response.statusCode}',
        );
      }
    } catch (e) {
      state = state.copyWith(
        status: VerificationStatus.failed,
        errorMessage: 'SSE 연결 오류: $e',
      );
    }
  }

  // SSE 라인 처리
  void _processSSELine(String line) {
    _lastReceivedData = line;
    try {
      // 빈 라인은 이벤트의 끝을 의미
      if (line.isEmpty) {
        if (_isCollectingEvent) {
          _handleSSEEvent(_currentEvent, _currentId, _currentData);

          // 버퍼 초기화
          _currentEvent = '';
          _currentId = '';
          _currentData = '';
          _isCollectingEvent = false;
        }
        return;
      }
      
      // 주석 라인 무시
      if (line.startsWith(':')) return;

      // 'id' 라인 처리
      if (line.startsWith('id: ')) {
        _currentId = line.substring(3).trim();
        return;
      }

       // 'data:' 라인 처리
      if (line.startsWith('data:')) {
        final data = line.substring(5).trim();
        _currentData = _currentData.isEmpty ? data : '$_currentData\n$data';
        return;
      }
      
      // 그 외 라인은 이전 데이터 계속
      if (_isCollectingEvent) {
        _currentData = '$_currentData\n$line';
      }
    } catch (e) {
      print('SSE 라인 처리 오류: $e');
    }
  }

  // SSE 이벤트 처리
  void _handleSSEEvent(String eventType, String eventId, String eventData) {
    print('SSE 이벤트: type=$eventType, id=$eventId, data=$eventData');

    // sse 이벤트 처리
    if (eventType == 'sse') {
      if (eventData.contains('인증이 완료되었습니다')) {
        state = state.copyWith(status: VerificationStatus.verified);
      } else if (eventData.contains('인증 코드가 만료되었습니다')) {
        state = state.copyWith(status: VerificationStatus.expired);
      } else if (eventData.contains('인증이 실패되었습니다')) {
        state = state.copyWith(status: VerificationStatus.failed);
      } else if (eventData.contains('EventStream Created')) {
        state = state.copyWith(status: VerificationStatus.verifying);
      } else if (eventData.contains('EventStream Closed')) {
        state = state.copyWith(status: VerificationStatus.connectionClosed);
      }
    }
  }

  @override
  void dispose() {
    _sseSubscription?.cancel();
    _httpClient?.close();
    super.dispose();
  }
}

// 인증 상태 Provider
final identityVerificationProvider = StateNotifierProvider<IdentityVerificationNotifier, VerificationState>((ref) {
  final repository = ref.watch(authIdentityRepositoryProvider);
  return IdentityVerificationNotifier(repository: repository);
});