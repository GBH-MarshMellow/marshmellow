import 'dart:io';
import 'package:dio/dio.dart';
import 'package:marshmellow/data/datasources/remote/api_client.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_model.dart';
import 'package:marshmellow/data/models/cookie/portfolio/portfolio_category_model.dart';

class PortfolioApi {
  final ApiClient _apiClient;

  PortfolioApi(this._apiClient);

  // 포트폴리오 목록 조회
  Future<Map<String, dynamic>> getPortfolioList() async {
    final response = await _apiClient.get('/portfolio/list');
    return response.data;
  }

  // 포트폴리오 카테고리 목록 조회
  Future<Map<String, dynamic>> getPortfolioCategoryList() async {
    final response = await _apiClient.get('/portfolio/category-list');
    return response.data;
  }

  // 포트폴리오 등록
  Future<Map<String, dynamic>> createPortfolio({
    required dynamic file,
    required String portfolioMemo,
    required String fileName,
    required int portfolioCategoryPk,
  }) async {
    // 멀티파트 요청을 위한 FormData 생성
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(
        file.path,
        filename: file.path.split('/').last,
      ),
      'portfolioMemo': portfolioMemo,
      'fileName': fileName,
      'portfolioCategoryPk': portfolioCategoryPk,
    });

    final response = await _apiClient.post(
      '/portfolio',
      data: formData,
    );
    return response.data;
  }

// 포트폴리오 수정
  Future<Map<String, dynamic>> updatePortfolio({
    required int portfolioPk,
    File? file,
    String? portfolioMemo,
    String? fileName,
    required int portfolioCategoryPk,
  }) async {
    // 멀티파트 요청을 위한 FormData 생성
    final formData = FormData.fromMap({
      'portfolioPk': portfolioPk,
      'portfolioCategoryPk': portfolioCategoryPk,
    });

    // 파일 파라미터가 항상 존재해야 함 (빈 파일로라도)
    if (file != null) {
      formData.files.add(MapEntry(
        'file',
        await MultipartFile.fromFile(
          file.path,
          filename: file.path.split('/').last,
        ),
      ));
    } else {
      // 파일이 없을 경우 빈 데이터로 전송
      formData.fields.add(MapEntry('file', ''));
    }

    // 선택적 파라미터 추가 (항상 값을 보내야 함)
    formData.fields.add(MapEntry('portfolioMemo', portfolioMemo ?? ''));
    formData.fields.add(MapEntry('fileName', fileName ?? ''));

    try {
      print('Sending portfolio update request:');
      print(
          'portfolioPk: $portfolioPk, portfolioCategoryPk: $portfolioCategoryPk');
      print('fileName: $fileName, portfolioMemo: $portfolioMemo');
      print('file updated: ${file != null}');

      final response = await _apiClient.patch(
        '/portfolio',
        data: formData,
      );

      print('Raw API response: ${response.data}');
      print('Response type: ${response.data.runtimeType}');

      // 응답이 문자열인 경우 처리
      if (response.data is String) {
        final message = response.data.toString().isNotEmpty
            ? response.data.toString()
            : '서버에서 오류가 발생했습니다';
        return {'code': 500, 'message': message, 'data': null};
      }

      // 응답이 비어있거나 null인 경우
      if (response.data == null) {
        return {'code': 500, 'message': '서버 응답이 없습니다', 'data': null};
      }

      // 객체인 경우 그대로 반환
      return response.data;
    } catch (e) {
      print('API error during update: $e');
      return {'code': 500, 'message': e.toString(), 'data': null};
    }
  }

// 포트폴리오 카테고리 등록
  Future<Map<String, dynamic>> createPortfolioCategory({
    required String categoryName,
    required String categoryMemo,
  }) async {
    print('🔍 API: 카테고리 생성 요청 - 이름: $categoryName, 메모: $categoryMemo');

    try {
      final data = {
        'categoryName': categoryName,
        'categoryMemo': categoryMemo,
      };

      print('📤 API: 요청 데이터 - $data');

      final response = await _apiClient.post(
        '/portfolio/category',
        data: data,
      );

      print('📥 API: 응답 상태 코드 - ${response.statusCode}');
      print('📥 API: 응답 데이터 - ${response.data}');

      return response.data;
    } catch (e) {
      print('❌ API: 카테고리 생성 오류 - $e');
      rethrow;
    }
  }

  // 포트폴리오 카테고리 삭제 메서드
  Future<Map<String, dynamic>> deletePortfolioCategories({
    required List<int> portfolioCategoryPkList,
  }) async {
    final response = await _apiClient.delete(
      '/portfolio/category-list',
      data: {
        'portfolioCategoryPkList': portfolioCategoryPkList,
      },
    );
    return response.data;
  }

// 포트폴리오 목록 삭제 메서드
  Future<Map<String, dynamic>> deletePortfolios({
    required List<int> portfolioPkList,
  }) async {
    final response = await _apiClient.delete(
      '/portfolio/list',
      data: {
        'portfolioPkList': portfolioPkList,
      },
    );
    return response.data;
  }
}
