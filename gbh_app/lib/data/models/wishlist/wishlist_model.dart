/*
  위시리스트 API 모델
*/

class WishlistResponse {
  final int code;
  final String message;
  final dynamic data;

  WishlistResponse({
    required this.code,
    required this.message,
    required this.data
  });

  factory WishlistResponse.fromJson(Map<String, dynamic> json) {
    return WishlistResponse(
      code: json['code'], 
      message: json['message'], 
      data: json['data']
    );
  }
}

class WishlistData {
  final String message;
  final List<Wishlist> wishlist;

  WishlistData({
    required this.message,
    required this.wishlist,
  });

  factory WishlistData.fromJson(Map<String, dynamic> json) {
    return WishlistData(
      message: json['message'], 
      wishlist: (json['wishlist'] as List)
          .map((item) => Wishlist.fromJson(item))
          .toList(),
    );
  }
}

class WishlistCreationResponse {
  final String message;
  final int wishlistPk;
  final String productNickname;
  final String productName;
  final int productPrice;
  final String? productImageUrl;
  final String? productUrl;
  final String isSelected;
  final String isCompleted;

  WishlistCreationResponse({
    required this.message,
    required this.wishlistPk,
    required this.productNickname,
    required this.productName,
    required this.productPrice,
    this.productImageUrl,
    this.productUrl,
    this.isSelected = "N",
    this.isCompleted = "N",
  });

  factory WishlistCreationResponse.fromJson(Map<String, dynamic> json) {
    return WishlistCreationResponse(
      message: json['message'] ?? '',
      wishlistPk: json['wishlistPk'], 
      productNickname: json['productNickname'], 
      productName: json['productName'], 
      productPrice: json['productPrice'],
      productImageUrl: json['productImageUrl'],
      productUrl: json['productUrl'],
      isSelected: json['isSelected'] ?? 'N',
      isCompleted: json['isCompleted'] ?? 'N',
    );
  }
}

class WishlistDetailResponse {
  final int wishlistPk;
  final String productNickname;
  final String productName;
  final int productPrice;
  final int achievePrice;
  final String? productImageUrl;
  final String? productUrl;
  final String isSelected;
  final String isCompleted;

  WishlistDetailResponse({
    required this.wishlistPk,
    required this.productNickname,
    required this.productName,
    required this.productPrice,
    required this.achievePrice,
    this.productImageUrl,
    this.productUrl,
    required this.isSelected,
    required this.isCompleted,
  });

  factory WishlistDetailResponse.fromJson(Map<String, dynamic> json) {
    return WishlistDetailResponse(
      wishlistPk: json['wishlistPk'],
      productNickname: json['productNickname'],
      productName: json['productName'],
      productPrice: json['productPrice'],
      achievePrice: json['achievePrice'],
      productImageUrl: json['productImageUrl'],
      productUrl: json['productUrl'],
      isSelected: json['isSelected'],
      isCompleted: json['isCompleted'],
    );
  }
}

class WishlistUpdateResponse {
  final String message;
  final int wishlistPk;
  final String oldNickname;
  final String newNickname;
  final String oldProductName;
  final String newProductName;
  final int oldProductPrice;
  final int newProductPrice;
  final String? oldProductImageUrl;
  final String? newProductImageUrl;
  final String? oldProductUrl;
  final String? newProductUrl;

  WishlistUpdateResponse({
    required this.message,
    required this.wishlistPk,
    required this.oldNickname,
    required this.newNickname,
    required this.oldProductName,
    required this.newProductName,
    required this.oldProductPrice,
    required this.newProductPrice,
    this.oldProductImageUrl,
    this.newProductImageUrl,
    this.oldProductUrl,
    this.newProductUrl,
  });

  factory WishlistUpdateResponse.fromJson(Map<String, dynamic> json) {
    return WishlistUpdateResponse(
      message: json['message'],
      wishlistPk: json['wishlistPk'],
      oldNickname: json['oldNickname'],
      newNickname: json['newNickname'],
      oldProductName: json['oldProductName'],
      newProductName: json['newProductName'],
      oldProductPrice: json['oldProductPrice'],
      newProductPrice: json['newProductPrice'],
      oldProductImageUrl: json['oldProductImageUrl'],
      newProductImageUrl: json['newProductImageUrl'],
      oldProductUrl: json['oldProductUrl'],
      newProductUrl: json['newProductUrl'],
    );
  }
}

class WishlistDeleteResponse {
  final String message;
  final int deleteWishlistPk;

  WishlistDeleteResponse({
    required this.message,
    required this.deleteWishlistPk,
  });

  factory WishlistDeleteResponse.fromJson(Map<String, dynamic> json) {
    return WishlistDeleteResponse(
      message: json['message'],
      deleteWishlistPk: json['deleteWishlistPk'],
    );
  }
}

class Wishlist {
  final int wishlistPk;
  final String productNickname;
  final String productName;
  final int productPrice;
  final String? productImageUrl;
  final String? productUrl;
  final String isSelected;
  final String isCompleted;

  Wishlist({
    required this.wishlistPk,
    required this.productNickname,
    required this.productName,
    required this.productPrice,
    this.productImageUrl,
    this.productUrl,
    required this.isSelected,
    required this.isCompleted,
  });

  factory Wishlist.fromJson(Map<String, dynamic> json) {
    return Wishlist(
      wishlistPk: json['wishlistPk'],
      productNickname: json['productNickname'],
      productName: json['productName'],
      productPrice: json['productPrice'],
      productImageUrl: json['productImageUrl'],
      productUrl: json['productUrl'],
      isSelected: json['isSelected'],
      isCompleted: json['isCompleted'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'wishlistPk': wishlistPk,
      'productNickname': productNickname,
      'productName': productName,
      'productPrice': productPrice,
      'productImageUrl': productImageUrl,
      'productUrl': productUrl,
      'isSelected': isSelected,
      'isCompleted': isCompleted,
    };
  }

  // 편의를 위한 copyWith 메서드
  Wishlist copyWith({
    int? wishlistPk,
    String? productNickname,
    String? productName,
    int? productPrice,
    String? productImageUrl,
    String? productUrl,
    String? isSelected,
    String? isCompleted,
  }) {
    return Wishlist(
      wishlistPk: wishlistPk ?? this.wishlistPk,
      productNickname: productNickname ?? this.productNickname,
      productName: productName ?? this.productName,
      productPrice: productPrice ?? this.productPrice,
      productImageUrl: productImageUrl ?? this.productImageUrl,
      productUrl: productUrl ?? this.productUrl,
      isSelected: isSelected ?? this.isSelected,
      isCompleted: isCompleted ?? this.isCompleted,
    );
  }
}