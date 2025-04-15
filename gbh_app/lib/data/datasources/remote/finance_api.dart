import 'dart:convert';

import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/api_client.dart';
import 'package:marshmellow/data/models/finance/transfer_model.dart';
import 'package:marshmellow/data/models/finance/withdrawal_account_model.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
//detail
import 'package:marshmellow/data/models/finance/asset_response_model.dart';
import 'package:marshmellow/data/models/finance/detail/card_detail_model.dart';
import 'package:marshmellow/data/models/finance/detail/demand_detail_model.dart';
import 'package:marshmellow/data/models/finance/detail/deposit_detail_model.dart';
import 'package:marshmellow/data/models/finance/detail/loan_detail_model.dart';
import 'package:marshmellow/data/models/finance/detail/saving_detail_model.dart';

// API 정의
class FinanceApi {
  final ApiClient _apiClient;
  
  FinanceApi(this._apiClient); // 의존성 주입
  
  Future<AssetResponseModel> getAssetInfo() async {
    final response = await _apiClient.get('/asset');
    
    final jsonString = jsonEncode(response.data);
    // 1000자씩 나누어 출력
    for (int i = 0; i < jsonString.length; i += 1000) {
      int end = (i + 1000 < jsonString.length) ? i + 1000 : jsonString.length;
      print('Response part ${i ~/ 1000 + 1}: ${jsonString.substring(i, end)}');
    }
    
    return AssetResponseModel.fromJson(response.data);
  }

  // 입출금 내역조회
  Future<DemandDetailResponse> getDemandAccountTransactions({
    required String accountNo,
    required String startDate,
    required String endDate,
    String transactionType = 'A',
    String? orderByType,
  }) async {
    final data = {
      'accountNo': accountNo,
      'startDate': startDate,
      'endDate': endDate,
      'transactionType': transactionType,
    };
    
    // orderByType이 null이 아닌 경우에만 추가
    if (orderByType != null) {
      data['orderByType'] = orderByType;
    }

    final response = await _apiClient.getWithBody('/asset/deposit-demand-transaction', data: data);
    return DemandDetailResponse.fromJson(response.data);
  }

  // 예금 조회 
  Future<DepositDetailResponse> getDepositPayment({
    required String accountNo,
  }) async {
    final data = {
      'accountNo': accountNo,
    };

    final response = await _apiClient.getWithBody('/asset/deposit-payment', data: data);
    return DepositDetailResponse.fromJson(response.data);
  }
  
  // 적금 납입 회차 조회 API
  Future<SavingDetailResponse> getSavingAccountPayments({
    required String accountNo,
  }) async {
    final data = {
      'accountNo': accountNo,
    };

    final response = await _apiClient.getWithBody('/asset/savings-payment', data: data);
    return SavingDetailResponse.fromJson(response.data);
  }

  // 대출 조회 
  Future<LoanDetailResponse> getLoanPaymentDetails({
    required String accountNo,
  }) async {
    final data = {
      'accountNo': accountNo,
    };

    try {
      final response = await _apiClient.getWithBody('/asset/loan-payment', data: data);
      // print("API 응답: $response.data"); // 디버깅용 로그 추가
      return LoanDetailResponse.fromJson(response.data);
    } catch (e) {
      print("API 에러: $e"); // 에러 로그 추가
      rethrow;
    }
  }

Future<CardDetailResponse> getCardTransactions({
  required String cardNo,
  required String cvc,
  required String startDate,
  required String endDate,
}) async {
  final data = {
    'cardNo': cardNo,
    'cvc': cvc,
    'startDate': startDate,
    'endDate': endDate,
  };

  try {
    final response = await _apiClient.getWithBody('/asset/card-transaction', data: data);
    
    // response.data가 null이 아닌지 확인
    if (response.data == null) {
      throw Exception('응답 데이터가 null입니다');
    }
    
    // 응답 타입 확인 로직
    if (response.data is String) {
      try {
        final Map<String, dynamic> parsedData = jsonDecode(response.data);
        return CardDetailResponse.fromJson(parsedData);
      } catch (e) {
        print('JSON 파싱 오류: $e');
        throw Exception('응답 데이터 파싱 실패: $e');
      }
    }
    
    return CardDetailResponse.fromJson(response.data);
  } catch (e) {
    print('카드 거래내역 조회 실패: $e');
    rethrow;
  }
}

  // 출금계좌 목록 조회
  Future<WithdrawalAccountResponse> getWithdrawalAccounts() async {
    final response = await _apiClient.getWithBody('/asset/withdrawal-account');
    // print('Withdrawal accounts response*****************: $response');
    return WithdrawalAccountResponse.fromJson(response.data);
  }

  // 계좌 인증 발송 (1원 송금)
  Future<AccountAuthResponse> sendAccountAuth({
    required String accountNo,
  }) async {
    final data = {
      'accountNo': accountNo,
    };
    
    final response = await _apiClient.post('/asset/open-account-auth', data: data);
    return AccountAuthResponse.fromJson(response.data);
  }

  // 계좌 인증 검증
  Future<AccountAuthVerifyResponse> verifyAccountAuth({
    required String accountNo,
    required String authCode,
  }) async {
    final data = {
      'accountNo': accountNo,
      'authCode': authCode,
      // 'authCode': authCode.toString(), // 명시적으로 문자열 변환
    };
    final response = await _apiClient.post('/asset/check-account-auth', data: data);
    return AccountAuthVerifyResponse.fromJson(response.data);
  }

  // 계좌 송금
  Future<TransferResponse> transferMoney(TransferRequest request) async {
    final response = await _apiClient.post('/asset/account-transfer', data: request.toJson());
    return TransferResponse.fromJson(response.data);
  }

}

// FinanceApi 프로바이더 정의
final financeApiProvider = Provider<FinanceApi>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  return FinanceApi(apiClient);
});