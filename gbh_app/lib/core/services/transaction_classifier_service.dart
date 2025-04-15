import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:marshmellow/data/datasources/remote/ledger_api.dart';
import 'package:marshmellow/data/datasources/remote/auth_api.dart';
import 'package:marshmellow/data/models/ledger/category/transactions.dart';
import 'package:marshmellow/di/providers/api_providers.dart';
import 'package:flutter/foundation.dart';

class TransactionSyncService {
  final LedgerApi _ledgerApi;
  final AuthApi _authApi;

  TransactionSyncService(this._ledgerApi, this._authApi);

  /// 동기화되지 않은 거래 내역 조회
  Future<List<Map<String, dynamic>>> fetchUnsyncedTransactions() async {
    try {
      final response = await _ledgerApi.getUnsyncedTransactions();

      if (kDebugMode) {
        print('📋 미동기화 거래 내역 조회 결과: $response');
      }

      // 응답 구조 변경: householdList를 직접 반환
      final householdList = response['householdList'] ?? [];
      return List<Map<String, dynamic>>.from(householdList);
    } catch (e) {
      if (kDebugMode) {
        print('❌ 동기화되지 않은 거래 내역 조회 실패: $e');
      }
      return [];
    }
  }

  /// AI를 통한 카테고리 분류
  Future<Map<String, String>> _classifyTransactionCategories(
      List<String> tradeNames) async {
    try {
      // 중복 제거
      final uniqueTradeNames = tradeNames.toSet().toList();

      if (kDebugMode) {
        print('🔍 카테고리 분류 요청: $uniqueTradeNames');
      }

      // 카테고리 분류 API 호출
      final Map<String, String> rawCategories =
          await _ledgerApi.classifyTransactionCategories(uniqueTradeNames);

      if (kDebugMode) {
        print('🔍 카테고리 분류 원본 결과: $rawCategories');
      }

      // 결과가 비어있거나 null인 경우 처리
      if (rawCategories.isEmpty) {
        if (kDebugMode) {
          print('⚠️ 카테고리 분류 결과가 비어있습니다. 기본값 사용.');
        }
        return {for (var name in tradeNames) name: '기타'};
      }

      // 요청한 이름에 대한 결과만 추출
      final categories = <String, String>{};
      for (var name in tradeNames) {
        if (rawCategories.containsKey(name)) {
          categories[name] = rawCategories[name]!;
        } else {
          categories[name] = '기타';
        }
      }

      if (kDebugMode) {
        print('📋 카테고리 분류 최종 결과: $categories');
      }

      return categories;
    } catch (e) {
      if (kDebugMode) {
        print('❌ 카테고리 분류 실패: $e');
      }
      // 실패 시 모든 트랜잭션을 '기타' 카테고리로 분류
      return {for (var name in tradeNames) name: '기타'};
    }
  }

  /// 거래 내역 동기화
  Future<SyncResult> syncTransactions(
      List<Map<String, dynamic>> transactions) async {
    try {
      if (transactions.isEmpty) {
        return SyncResult(
          success: true,
          totalTransactions: 0,
          message: '동기화할 거래 내역 없음',
        );
      }

      // 1. 트레이드명 추출
      final tradeNames = transactions
          .map((t) => t['tradeName'] as String)
          .where((name) => name != null && name.isNotEmpty)
          .toList();

      // 2. 카테고리 분류
      final categories = await _classifyTransactionCategories(tradeNames);

      // 3. 요청 데이터 형식에 맞게 트랜잭션 변환
      final formattedTransactions = transactions.map((transaction) {
        final tradeName = transaction['tradeName'] as String;

        // 카테고리 결정 (API 응답 또는 기본값)
        final category = categories[tradeName] ?? '기타';

        // user 객체에서 필요한 필드만 추출하고 문자열로 변환
        final userMap = transaction['user'] as Map<String, dynamic>;
        final filteredUser = {
          'userPk': userMap['userPk'],
          'userName': userMap['userName']?.toString() ?? '',
          'userEmail': userMap['userEmail']?.toString() ?? '',
          'phoneNumber': userMap['phoneNumber']?.toString() ?? '',
          'birth': userMap['birth']?.toString() ?? '',
          'gender': userMap['gender']?.toString() ?? '',
          'pin': userMap['pin']?.toString() ?? '',
          'userKey': userMap['userKey']?.toString() ?? '',
          'characterImageUrl': userMap['characterImageUrl'],
          'budgetFeature': userMap['budgetFeature'],
          'budgetAlarmTime': userMap['budgetAlarmTime'],
          'createdAt': userMap['createdAt']?.toString() ?? '',
        };

        // 문자열 값을 문자열로 강제 변환하여 API 요구사항에 맞게 데이터 구성
        final formattedTransaction = {
          'householdPk': transaction['householdPk'] ?? 0,
          'tradeName': tradeName.toString(),
          'tradeDate': transaction['tradeDate']?.toString() ?? '',
          'tradeTime': transaction['tradeTime']?.toString() ?? '',
          'householdAmount': transaction['householdAmount'] is int
              ? transaction['householdAmount']
              : int.tryParse(
                      transaction['householdAmount']?.toString() ?? '0') ??
                  0,
          'householdMemo': transaction['householdMemo'],
          'paymentMethod': transaction['paymentMethod']?.toString() ?? '',
          'paymentCancelYn': transaction['paymentCancelYn']?.toString() ?? 'N',
          'exceptedBudgetYn':
              transaction['exceptedBudgetYn']?.toString() ?? 'N',
          'user': filteredUser,
          'category': category,
          'householdClassificationCategory':
              transaction['householdClassificationCategory']?.toString() ??
                  'WITHDRAWAL',
        };

        return formattedTransaction;
      }).toList();

      // 변환된 거래 내역 로깅
      if (kDebugMode && formattedTransactions.isNotEmpty) {
        final firstItem = formattedTransactions.first;
        print('📝 변환된 거래 내역 (첫 번째 항목): $firstItem');

        // JSON 형태로 직렬화하여 형식 검증
        final jsonSample = jsonEncode(firstItem);
        print('📝 JSON으로 직렬화: $jsonSample');
      }

      // 4. 요청 데이터 준비 및 직렬화
      final requestData = {'transactionList': formattedTransactions};

      // 5. 가계부에 거래 내역 등록
      final registrationResult =
          await _ledgerApi.registerTransactions(requestData);

      if (kDebugMode) {
        print('✅ 가계부 거래 등록 결과: $registrationResult');
      }

      return SyncResult(
        success: true,
        totalTransactions: formattedTransactions.length,
        message: '거래 내역 동기화 완료 (${formattedTransactions.length}건)',
      );
    } catch (e) {
      if (kDebugMode) {
        print('❌ 거래 내역 동기화 실패: $e');
      }
      return SyncResult(
        success: false,
        totalTransactions: transactions.length,
        message: '거래 내역 동기화 중 오류 발생: $e',
      );
    }
  }

  /// 전체 동기화 워크플로우
  Future<SyncResult> performFullSync() async {
    try {
      // 1. 동기화되지 않은 거래 내역 조회
      final unsyncedTransactions = await fetchUnsyncedTransactions();

      if (unsyncedTransactions.isEmpty) {
        return SyncResult(
          success: true,
          totalTransactions: 0,
          message: '동기화할 거래 내역 없음',
        );
      }

      // 2. 거래 내역 동기화
      return await syncTransactions(unsyncedTransactions);
    } catch (e) {
      if (kDebugMode) {
        print('❌ 전체 동기화 과정 중 오류 발생: $e');
      }
      return SyncResult(
        success: false,
        totalTransactions: 0,
        message: '동기화 중 예상치 못한 오류 발생: $e',
      );
    }
  }

  /// 미분류 내역 확인
  Future<bool> hasUnsortedTransactions() async {
    final transactions = await fetchUnsyncedTransactions();
    return transactions.isNotEmpty;
  }
}

/// 동기화 결과를 나타내는 클래스
class SyncResult {
  final bool success;
  final int totalTransactions;
  final String message;

  SyncResult({
    required this.success,
    required this.totalTransactions,
    required this.message,
  });
}

// Riverpod Provider
final transactionSyncServiceProvider = Provider<TransactionSyncService>((ref) {
  final apiClient = ref.watch(apiClientProvider);
  final authApi = ref.watch(authApiProvider);
  final ledgerApi = LedgerApi(apiClient);
  return TransactionSyncService(ledgerApi, authApi);
});
