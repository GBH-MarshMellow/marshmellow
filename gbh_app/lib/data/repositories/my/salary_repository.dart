import 'package:dio/dio.dart';
import 'package:marshmellow/data/datasources/remote/my/salary_api.dart';
import 'package:marshmellow/data/models/my/salary_model.dart';

class MySalaryRepository {
  final MySalaryApi _mySalaryApi;

  MySalaryRepository(this._mySalaryApi);

  // 입출금계좌목록조회
  Future<List<AccountModel>> getAccountList() async {
    try {
      final response = await _mySalaryApi.getAccountList();
      final accountListResponse = AccountListResponse.fromJson(response.data);

      if (accountListResponse.code == 200 && accountListResponse.data != null) {
        return accountListResponse.data!.accountList;
      } else {
        throw Exception('계좌 목록 조회 실패: ${accountListResponse.message}');
      }
    } catch (e) {
      if (e is DioException) {
        throw Exception('네트워크 오류: ${e.message}');
      }
      rethrow;
    }
  }

  // 입금내역조회
  Future<List<DepositModel>> getDepositList(String accountNo) async {
    try {
      final response = await _mySalaryApi.getDepositList(accountNo);
      final depositListResponse = DepositListResponse.fromJson(response.data);

      if (depositListResponse.code == 200 && depositListResponse.data != null) {
        return depositListResponse.data!.depositList;
      } else {
        throw Exception('입금 내역 조회 실패: ${depositListResponse.message}');
      }
    } catch (e) {
      if (e is DioException) {
        throw Exception('네트워크 오류: ${e.message}');
      }
      rethrow;
    }
  }

  // 월급등록
  Future<bool> registerSalary(int salary, int date) async {
    try {
      final response = await _mySalaryApi.registerSalary(salary, date);
      final salaryResponse = SalaryResponse.fromJson(response.data);

      if (salaryResponse.code == 200 &&
          salaryResponse.data != null &&
          salaryResponse.data!.message == 'SUCCESS') {
        return true;
      } else {
        throw Exception('월급 등록 실패: ${salaryResponse.message}');
      }
    } catch (e) {
      if (e is DioException) {
        throw Exception('네트워크 오류: ${e.message}');
      }
      rethrow;
    }
  }

  // 월급수정
  Future<bool> updateSalary(int salary, int date) async {
    try {
      final response = await _mySalaryApi.updateSalary(salary, date);
      final salaryResponse = SalaryResponse.fromJson(response.data);

      if (salaryResponse.code == 200 &&
          salaryResponse.data != null &&
          salaryResponse.data!.message == 'SUCCESS') {
        return true;
      } else {
        throw Exception('월급 수정 실패: ${salaryResponse.message}');
      }
    } catch (e) {
      if (e is DioException) {
        throw Exception('네트워크 오류: ${e.message}');
      }
      rethrow;
    }
  }

  // 월급일 조회
  Future<int> getSalaryDay() async {
    try {
      final response = await _mySalaryApi.getSalaryDay();
      return response;
    } catch (e) {
      if (e is DioException) {
        throw Exception('Network error: ${e.message}');
      }
      rethrow;
    }
  }
}
