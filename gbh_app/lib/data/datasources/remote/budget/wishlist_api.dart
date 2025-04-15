import 'dart:io';
import 'package:dio/dio.dart';
import 'package:http_parser/http_parser.dart';
import 'package:path/path.dart' as path;
import 'package:marshmellow/data/datasources/remote/api_client.dart';

class WishlistApi {
  final ApiClient _apiClient;
  
  WishlistApi(this._apiClient);

  // 위시리스트 생성 - 이미지 파일 지원 추가
  Future<Map<String, dynamic>> createWishlist({
    required String productNickname,
    required String productName,
    required int productPrice,
    required String productUrl,
    File? imageFile,
  }) async {
    // FormData 형식으로 변환
    FormData formData = FormData.fromMap({
      'productNickname': productNickname,
      'productName': productName,
      'productPrice': productPrice.toString(),
      'productUrl': productUrl,
    });
    
    // 이미지 파일 추가
    if (imageFile != null) {
      final String fileName = path.basename(imageFile.path);
      final String fileExtension = path.extension(fileName).toLowerCase().substring(1);
      
      formData.files.add(MapEntry(
        'file',
        await MultipartFile.fromFile(
          imageFile.path,
          filename: fileName,
          contentType: MediaType('image', fileExtension),
        ),
      ));
    }

    final response = await _apiClient.post('/mm/wishlist', data: formData);
    return response.data;
  }

  // 위시 리스트 전체 조회
  Future<Map<String, dynamic>> getWishlists() async {
    final response = await _apiClient.get('/mm/wishlist');
    return response.data;
  }

  // 위시 리스트 상세 조회
  Future<Map<String, dynamic>> getWishlistDetail(int wishlistPk) async {
    final response = await _apiClient.get('/mm/wishlist/detail/$wishlistPk');
    return response.data;
  }
  
  // 위시 리스트 수정 - 이미지 파일 지원 추가
  Future<Map<String, dynamic>> updateWishlist({
  required int wishlistPk,
  String? productNickname,
  String? productName,
  int? productPrice,
  String? productImageUrl,  // 추가된 파라미터
  String? productUrl,
  String? isSelected,
  String? isCompleted,
  File? imageFile,
  }) async {
    // 업데이트할 데이터
    Map<String, dynamic> data = {};
    
    // null이 아닌 값만 추가
    if (productNickname != null) data['productNickname'] = productNickname;
    if (productName != null) data['productName'] = productName;
    if (productPrice != null) data['productPrice'] = productPrice.toString();
    if (productImageUrl != null) data['productImageUrl'] = productImageUrl;  // 추가됨
    if (productUrl != null) data['productUrl'] = productUrl;
    if (isSelected != null) data['isSelected'] = isSelected;
    if (isCompleted != null) data['isCompleted'] = isCompleted;

    // 이미지 파일이 있는 경우 FormData로 변환
    if (imageFile != null) {
      FormData formData = FormData.fromMap(data);
      
      final String fileName = path.basename(imageFile.path);
      final String fileExtension = path.extension(fileName).toLowerCase().substring(1);
      
      formData.files.add(MapEntry(
        'file',
        await MultipartFile.fromFile(
          imageFile.path,
          filename: fileName,
          contentType: MediaType('image', fileExtension),
        ),
      ));
      
      final response = await _apiClient.put('/mm/wishlist/detail/$wishlistPk', data: formData);
      return response.data;
    } else {
      // 이미지가 없는 경우 일반 PUT 요청
      final response = await _apiClient.put('/mm/wishlist/detail/$wishlistPk', data: data);
      return response.data;
    }
  }

  // 위시리스트 삭제
  Future<Map<String, dynamic>> deleteWishlist(int wishlistPk) async {
    final response = await _apiClient.delete('/mm/wishlist/detail/$wishlistPk');
    return response.data;
  }

  // 위시리스트 크롤링
  Future<Map<String, dynamic>> crawlProductUrl(String url) async {
  try {
    print("🐥🐥크롤링 요청 URL: $url");
    final data = {'url': url};
    print("🐥🐥요청 데이터: $data");
    
    final response = await _apiClient.post('/mm/wishlist/jsoup', data: data);
    print("🐥🐥응답 상태 코드: ${response.statusCode}");
    print("🐥🐥응답 데이터: ${response.data}");
    
    return response.data;
  } catch (e) {
    print("🐥🐥크롤링 API 오류: $e");
    if (e is DioException) {
      print("🐥🐥상태 코드: ${e.response?.statusCode}");
      print("🐥🐥응답 데이터: ${e.response?.data}");
      print("🐥🐥요청 데이터: ${e.requestOptions.data}");
      print("🐥🐥요청 헤더: ${e.requestOptions.headers}");
    }
    rethrow;
  }
}
}

class WishApi {
  final ApiClient _apiClient;

  WishApi(this._apiClient);

  // 현재 진행중인 wish 조회
  Future<Map<String, dynamic>> getCurrentWish() async {
    final response = await _apiClient.get('/mm/wishlist/detail');
    return response.data;
  }

  // 특정 wish 상세 조회
  Future<Map<String, dynamic>> getWishDetail(int wishPk) async {
    final response = await _apiClient.get('/mm/wish/detail/$wishPk');
    return response.data;
  }

  // wish 생성
  Future<Map<String, dynamic>> selectWish(int wishPk, String isSelected) async {
    final data = {'isSelected' : isSelected};
    final response = await _apiClient.post('/mm/wishlist/detail/$wishPk', data: data);
    return response.data;
  }

  // 자동이체 등록
  Future<Map<String, dynamic>> registerAutoTransaction({
    required String withdrawalAccountNo, // 출금 계좌번호
    required String depositAccountNo,    // 입금 계좌번호
    required String dueDate,             // 자동이체 종료일
    required int transactionBalance,  // 자동이체 금액
    required int wishListPk,             // 위시리스트 고유 번호
    required int userPk,                 // 회원 고유번호
  }) async {
    final data = {
      'withdrawalAccountNo': withdrawalAccountNo,
      'depositAccountNo': depositAccountNo,
      'dueDate': dueDate,
      'transactionBalance': transactionBalance,
      'wishListPk': wishListPk,
      'userPk': userPk,
    };
    
    final response = await _apiClient.post('/household/transaction-data', data: data);
    return response.data;
  }

  // 입출금 계좌 목록 조회
  Future<Map<String, dynamic>> getDemDepList() async {
    final response = await _apiClient.get('/auto-transaction/demand-deposit-list');
    return response.data;
  }
}