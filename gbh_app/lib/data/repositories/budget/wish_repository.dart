import 'package:marshmellow/data/datasources/remote/budget/wishlist_api.dart';
import 'package:marshmellow/data/models/wishlist/wish_model.dart';

class WishRepository {
  final WishApi _wishApi;

  WishRepository(this._wishApi);

  // 현재 진행 중인 wish 조회
  Future<WishDetail?> getCurrentWish() async {
    try {
      print('Repository: Fetching current wish...');
      final response = await _wishApi.getCurrentWish();
      final wishResponse = WishResponse.fromJson(response);

      if (wishResponse.code == 200 && wishResponse.data != null) {
        return WishDetail.fromJson(wishResponse.data);
      } else {
        // 404 등 정상 응답이지만 데이터 없는 경우
        print('Repository: No wish data found or non-200 code');
        return null;
      }
    } catch (e) {
      throw Exception('현재 진행 중인 wish를 불러오는 중 오류가 발생했습니다: $e');
    }
  }

  // 위시 선택 (isSelected 상태 변경)
  Future<WishSelectionResponse> selectWish(int wishPk, String isSelected) async {
    try {
      final response = await _wishApi.selectWish(wishPk, isSelected);
      return WishSelectionResponse.fromJson(response);
    } catch (e) {
      throw Exception('위시 선택 중 오류가 발생했습니다 : $e');
    }
  }

  // 자동이체 등록
  Future<AutoTransferResponse> registerAutoTransfer({
    required String withdrawalAccountNo,
    required String depositAccountNo,
    required String dueDate,
    required int transactionBalance,
    required int wishListPk,
    required int userPk,
  }) async {
    try {
      final response = await _wishApi.registerAutoTransaction(
        withdrawalAccountNo: withdrawalAccountNo,
        depositAccountNo: depositAccountNo,
        dueDate: dueDate,
        transactionBalance: transactionBalance,
        wishListPk: wishListPk,
        userPk: userPk,
      );
      return AutoTransferResponse.fromJson(response);
    } catch (e) {
      throw Exception('자동이체 등록 중 오류가 발생했습니다: $e');
    }
  }

  // 입출금 계좌 목록 조회
  Future<DemandDepositResponse> getDemDepList() async {
    try {
      final response = await _wishApi.getDemDepList();
      return DemandDepositResponse.fromJson(response);
    } catch (e) {
      throw Exception('입출금 계좌 목록을 불러오는 중 오류가 발생했습니다 : $e');
    }
  }
}