import 'dart:io';
import 'package:marshmellow/data/datasources/remote/budget/wishlist_api.dart';
import 'package:marshmellow/data/models/wishlist/wishlist_model.dart';

class WishlistRepository {
  final WishlistApi _wishlistApi;

  WishlistRepository(this._wishlistApi);

  // 위시리스트 생성
  Future<WishlistCreationResponse> createWishlist({
    required String productNickname,
    required String productName,
    required int productPrice,
    required String productUrl,
    File? imageFile,
  }) async {
    final response = await _wishlistApi.createWishlist(
      productNickname: productNickname,
      productName: productName,
      productPrice: productPrice,
      productUrl: productUrl,
      imageFile: imageFile,
    );

    final wishlistResponse = WishlistResponse.fromJson(response);

    if (wishlistResponse.code == 200) {
      // 응답 형식에 맞게 파싱
      final responseData = wishlistResponse.data;
      return WishlistCreationResponse(
        message: responseData['message'] ?? '',
        wishlistPk: responseData['wishlistPk'],
        productNickname: responseData['productNickname'],
        productName: responseData['productName'],
        productPrice: responseData['productPrice'],
        productImageUrl: responseData['productImageUrl'],
        productUrl: responseData['productUrl'],
      );
    } else {
      throw Exception(wishlistResponse.message);
    }
  }

  // 위시리스트 조회
  Future<List<Wishlist>> getWishlists() async {
    final response = await _wishlistApi.getWishlists();
    final wishlistResponse = WishlistResponse.fromJson(response);

    if (wishlistResponse.code == 200) {
      final data = WishlistData.fromJson(wishlistResponse.data);
      return data.wishlist;
    } else {
      throw Exception(wishlistResponse.message);
    }
  }

  // 위시리스트 상세 조회
  Future<WishlistDetailResponse> getWishlistDetail(int wishlistPk) async {
    final response = await _wishlistApi.getWishlistDetail(wishlistPk);
    final wishlistResponse = WishlistResponse.fromJson(response);

    if (wishlistResponse.code == 200) {
      return WishlistDetailResponse.fromJson(wishlistResponse.data);
    } else {
      throw Exception(wishlistResponse.message);
    }
  }

  // 위시리스트 수정
  Future<WishlistUpdateResponse> updateWishlist({
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
    final response = await _wishlistApi.updateWishlist(
      wishlistPk: wishlistPk,
      productNickname: productNickname,
      productName: productName,
      productPrice: productPrice,
      productImageUrl: productImageUrl,  // API에 전달
      productUrl: productUrl,
      isSelected: isSelected,
      isCompleted: isCompleted,
      imageFile: imageFile,
    );

    final wishlistResponse = WishlistResponse.fromJson(response);

    if (wishlistResponse.code == 200) {
      // 응답 변경에 따른 처리
      final responseData = wishlistResponse.data;
      
      // 새 응답 형식에 맞게 변환
      return WishlistUpdateResponse(
        message: responseData['message'] ?? '',
        wishlistPk: responseData['wishlistPk'],
        oldNickname: responseData['oldNickname'] ?? responseData['productNickname'],
        newNickname: responseData['productNickname'],
        oldProductName: responseData['oldProductName'] ?? responseData['productName'],
        newProductName: responseData['productName'],
        oldProductPrice: responseData['oldProductPrice'] ?? responseData['productPrice'],
        newProductPrice: responseData['productPrice'],
        oldProductImageUrl: responseData['oldProductImageUrl'] ?? responseData['productImageUrl'],
        newProductImageUrl: responseData['productImageUrl'],
        oldProductUrl: responseData['oldProductUrl'] ?? responseData['productUrl'],
        newProductUrl: responseData['productUrl'],
      );
    } else {
      throw Exception(wishlistResponse.message);
    }
  }

  // 위시리스트 삭제
  Future<WishlistDeleteResponse> deleteWishlist(int wishlistPk) async {
    final response = await _wishlistApi.deleteWishlist(wishlistPk);
    final wishlistResponse = WishlistResponse.fromJson(response);

    if (wishlistResponse.code == 200) {
      return WishlistDeleteResponse.fromJson(wishlistResponse.data);
    } else {
      throw Exception(wishlistResponse.message);
    }
  }

  // 링크 크롤링
  Future<Map<String, dynamic>> crawlProductUrl(String url) async {
    final response = await _wishlistApi.crawlProductUrl(url);
    final wishlistResponse = WishlistResponse.fromJson(response);

    if (wishlistResponse.code == 200) {
      return wishlistResponse.data; 
    } else {
      throw Exception(wishlistResponse.message);
    }
  }
}