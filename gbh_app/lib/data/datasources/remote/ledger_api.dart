import 'package:marshmellow/data/datasources/remote/api_client.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/data/models/ledger/payment_method.dart';
import 'package:flutter/foundation.dart';
import 'dart:convert';

class LedgerApi {
  final ApiClient _apiClient;

  LedgerApi(this._apiClient);

  // 가계부 목록 조회 (기간별)
  // [startDate] 조회 시작일 (yyyyMMdd 형식)
  // [endDate] 조회 종료일 (yyyyMMdd 형식)
  Future<Map<String, dynamic>> getHouseholdList({
    required String startDate,
    required String endDate,
  }) async {
    try {
      final response = await _apiClient.getWithBody(
        '/household/list',
        data: {
          'startDate': startDate,
          'endDate': endDate,
        },
      );

      if (response.data['code'] == 200 && response.data['data'] != null) {
        final data = response.data['data'];

        // 총 수입/지출 데이터
        final totalIncome = data['totalIncome'] ?? 0;
        final totalExpenditure = data['totalExpenditure'] ?? 0;

        // 날짜별 가계부 목록 처리
        final householdList = data['householdList'] ?? [];

        // 날짜별로 그룹화된 트랜잭션 목록
        final Map<String, List<Transaction>> groupedTransactions = {};

        // 모든 트랜잭션 목록
        final List<Transaction> allTransactions = [];

        // householdList 순회하며 트랜잭션 변환
        for (var dateGroup in householdList) {
          final date = dateGroup['date'] as String;
          final list = dateGroup['list'] as List;

          groupedTransactions[date] = [];

          for (var item in list) {
            final transaction = Transaction.fromJson(item);
            groupedTransactions[date]!.add(transaction);
            allTransactions.add(transaction);
          }
        }

        // 결과 반환
        return {
          'totalIncome': totalIncome,
          'totalExpenditure': totalExpenditure,
          'groupedTransactions': groupedTransactions,
          'allTransactions': allTransactions,
        };
      }

      throw Exception('API 응답 에러: ${response['message']}');
    } catch (e) {
      throw Exception('가계부 목록을 가져오는데 실패했습니다: $e');
    }
  }

  // 가계부 상세 조회
  Future<Transaction> getHouseholdDetail(int householdPk) async {
    try {
      final response = await _apiClient.getWithBody(
        '/household',
        data: {
          'householdPk': householdPk,
        },
      );

      if (response.data['code'] == 200 && response.data['data'] != null) {
        final data = response.data['data'];
        return Transaction.fromJson(data);
      }

      throw Exception('API 응답 에러: ${response.data['message']}');
    } catch (e) {
      throw Exception('가계부 상세 정보를 가져오는데 실패했습니다: $e');
    }
  }

  // 검색 기능
  Future<Map<String, dynamic>> searchHousehold({
    required String startDate,
    required String endDate,
    required String keyword,
  }) async {
    try {
      final response = await _apiClient.getWithBody(
        '/household/search',
        data: {
          'startDate': startDate,
          'endDate': endDate,
          'keyword': keyword,
        },
      );

      if (response.data['code'] == 200 && response.data['data'] != null) {
        final data = response.data['data'];
        final List<Transaction> transactions = [];

        if (data['householdList'] != null) {
          // 날짜별 그룹을 순회
          for (var dateGroup in data['householdList']) {
            // 각 날짜 그룹 내의 트랜잭션 리스트 처리
            if (dateGroup['list'] != null && dateGroup['list'] is List) {
              for (var item in dateGroup['list']) {
                try {
                  // 트랜잭션 객체 생성
                  final transaction = Transaction.fromJson(item);
                  transactions.add(transaction);
                } catch (e) {
                  print('Error parsing transaction: $e');
                }
              }
            }
          }
        }

        return {
          'transactions': transactions,
        };
      }

      throw Exception('API 응답 에러: ${response.data['message']}');
    } catch (e) {
      throw Exception('가계부 검색에 실패했습니다: $e');
    }
  }

  // 가계부 삭제
  Future<Map<String, dynamic>> deleteHousehold({
    required int householdPk,
  }) async {
    try {
      final response = await _apiClient.delete(
        '/household',
        data: {
          'householdPk': householdPk,
        },
      );

      if (response.data['code'] == 200) {
        return response.data['data'];
      }

      throw Exception('API 응답 에러: ${response['message']}');
    } catch (e) {
      throw Exception('가계부 삭제에 실패했습니다: $e');
    }
  }

  // 가계부 수정
  Future<Transaction> updateHousehold({
    required int householdPk,
    int? householdAmount,
    String? householdMemo,
    String? exceptedBudgetYn,
    int? householdDetailCategoryPk,
  }) async {
    try {
      // API 요청 데이터 준비
      final Map<String, dynamic> requestData = {
        'householdPk': householdPk,
      };

      // 선택적 매개변수 추가
      if (householdAmount != null)
        requestData['householdAmount'] = householdAmount;
      if (householdMemo != null) requestData['householdMemo'] = householdMemo;
      if (exceptedBudgetYn != null)
        requestData['exceptedBudgetYn'] = exceptedBudgetYn;
      if (householdDetailCategoryPk != null)
        requestData['householdDetailCategoryPk'] = householdDetailCategoryPk;

      final response = await _apiClient.patch(
        '/household',
        data: requestData,
      );

      if (response.data['code'] == 200 && response.data['data'] != null) {
        final data = response.data['data'];
        return Transaction.fromJson(data);
      }

      throw Exception('API 응답 에러: ${response.data['message']}');
    } catch (e) {
      throw Exception('가계부 수정에 실패했습니다: $e');
    }
  }

  // 가계부 분류별 조회 API 호출
  Future<Map<String, dynamic>> getHouseholdFilter({
    required String startDate,
    required String endDate,
    required String classification, // 'DEPOSIT', 'WITHDRAWAL', 'TRANSFER'
  }) async {
    try {
      final response = await _apiClient.getWithBody(
        '/household/filter',
        data: {
          'startDate': startDate,
          'endDate': endDate,
          'classification': classification,
        },
      );

      if (response.data['code'] == 200 && response.data['data'] != null) {
        return response.data['data'];
      }

      throw Exception('API 응답 에러: ${response.data['message']}');
    } catch (e) {
      throw Exception('분류별 가계부 목록을 가져오는데 실패했습니다: $e');
    }
  }

  // 결제수단 목록 조회
  Future<PaymentMethodResponse> getPaymentMethods() async {
    try {
      final response = await _apiClient.get('/household/payment-method');

      if (response.data != null) {
        return PaymentMethodResponse.fromJson(response.data);
      }

      throw Exception('API 응답 에러: ${response.data['message']}');
    } catch (e) {
      throw Exception('결제수단 목록을 가져오는데 실패했습니다: $e');
    }
  }

  // 가계부 등록
  Future<Map<String, dynamic>> createHousehold({
    required String tradeName,
    required String tradeDate,
    required String tradeTime,
    required int householdAmount,
    String? householdMemo,
    required String paymentMethod,
    required String exceptedBudgetYn,
    required String householdClassification,
    required int householdDetailCategoryPk,
  }) async {
    try {
      final response = await _apiClient.post(
        '/household',
        data: {
          'tradeName': tradeName,
          'tradeDate': tradeDate,
          'tradeTime': tradeTime,
          'householdAmount': householdAmount,
          'householdMemo': householdMemo,
          'paymentMethod': paymentMethod,
          'exceptedBudgetYn': exceptedBudgetYn,
          'householdClassification': householdClassification,
          'householdDetailCategoryPk': householdDetailCategoryPk,
        },
      );

      if (response.data['code'] == 200 && response.data['data'] != null) {
        return response.data['data'];
      }

      throw Exception('API 응답 에러: ${response.data['message']}');
    } catch (e) {
      throw Exception('가계부 등록 API 호출 실패: $e');
    }
  }

  // 미분류 거래 내역 조회
  Future<Map<String, dynamic>> getUnsyncedTransactions() async {
    try {
      final response = await _apiClient.get('/household/transaction-data');

      if (response.data['code'] == 200 && response.data['data'] != null) {
        return response.data['data'];
      }

      throw Exception('API 응답 에러: ${response.data['message']}');
    } catch (e) {
      throw Exception('미동기화 거래 내역을 가져오는데 실패했습니다: $e');
    }
  }

// 거래 내역 일괄 등록
  Future<Map<String, dynamic>> registerTransactions(
      Map<String, dynamic> transactionData) async {
    try {
      if (kDebugMode) {
        print('📤 거래 내역 등록 API 호출: /household/household-list');

        // 데이터 구조 확인
        if (transactionData.containsKey('transactionList') &&
            transactionData['transactionList'] is List &&
            (transactionData['transactionList'] as List).isNotEmpty) {
          print(
              '📦 거래 내역 수: ${(transactionData['transactionList'] as List).length}');
          print(
              '📦 첫 번째 항목 예시: ${(transactionData['transactionList'] as List).first}');
        }
      }

      // 요청 데이터를 JSON 문자열로 직렬화
      final jsonString = jsonEncode(transactionData);

      if (kDebugMode) {
        print('📦 JSON 요청 데이터: $jsonString');
      }

      // API 호출 시 직렬화된 JSON 문자열을 사용
      final response = await _apiClient.post(
        '/household/household-list',
        data: jsonString,
      );

      if (kDebugMode) {
        print('📥 거래 내역 등록 API 응답 코드: ${response.statusCode}');
        print('📥 응답 데이터: ${response.data}');
      }

      if (response.data != null && response.data['code'] == 200) {
        return response.data['data'] ?? {};
      }

      throw Exception(
          'API 응답 에러: ${response.data?['message'] ?? '알 수 없는 오류'} (${response.data?['code'] ?? 'No code'})');
    } catch (e) {
      if (kDebugMode) {
        print('❌ 거래 내역 등록 API 호출 중 오류: $e');
      }
      throw Exception('거래 내역 등록에 실패했습니다: $e');
    }
  }

// AI 카테고리 분류
  Future<Map<String, String>> classifyTransactionCategories(
      List<String> tradeNames) async {
    try {
      // 디버그 로그
      if (kDebugMode) {
        print('카테고리 분류 API 호출: 경로=/mm/ai/category, 데이터=${{
          "tradeNames": tradeNames
        }}');
      }

      // API 호출
      final response = await _apiClient
          .post('/mm/ai/category', data: {'tradeNames': tradeNames});

      // 디버그 로그
      if (kDebugMode) {
        print(
            '카테고리 분류 API 응답: 상태코드=${response.statusCode}, 데이터=${response.data}');
      }

      // 응답 검증
      if (response.data == null) {
        throw Exception('API 응답이 없습니다.');
      }

      // 응답이 직접 Map 형태로 오는 경우 (로그에서 본 형태)
      if (response.data is Map) {
        final Map rawData = response.data;
        final Map<String, String> result = {};

        // 직접 Map 구조 처리
        for (var tradeName in tradeNames) {
          if (rawData.containsKey(tradeName)) {
            result[tradeName] = rawData[tradeName].toString();
          } else {
            result[tradeName] = '기타';
          }
        }

        if (kDebugMode) {
          print('처리된 카테고리 결과: $result');
        }

        return result;
      }

      // 응답 구조 또는 데이터 누락 시 기본값 처리
      return {for (var name in tradeNames) name: '미분류'};
    } catch (e) {
      if (kDebugMode) {
        print('카테고리 분류 API 호출 중 오류: $e');
      }
      return {for (var name in tradeNames) name: '미분류'};
    }
  }
}
