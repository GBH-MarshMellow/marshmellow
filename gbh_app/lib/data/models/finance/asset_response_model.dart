// 응답의 최상위 모델
import 'package:marshmellow/data/models/finance/account_models.dart';
import 'package:marshmellow/data/models/finance/card_model.dart';

class AssetResponseModel {
  final int code;
  final String message;
  final AssetData data;

  AssetResponseModel({
    required this.code,
    required this.message,
    required this.data,
  });

  factory AssetResponseModel.fromJson(Map<String, dynamic> json) {
    return AssetResponseModel(
      code: json['code'],
      message: json['message'],
      data: AssetData.fromJson(json['data']),
    );
  }
}

// 모든 자산 데이터를 포함하는 모델
class AssetData {
  //**********************************************
  //* 추가: IV 필드 추가
  //**********************************************
  final String? iv;  // 응답에 포함된 IV 추가
  final CardData cardData;
  final DemandDepositData demandDepositData;
  final LoanData loanData;
  final SavingsData savingsData;
  final DepositData depositData;

  AssetData({
    this.iv,  // IV 필드 추가
    required this.cardData,
    required this.demandDepositData,
    required this.loanData,
    required this.savingsData,
    required this.depositData,
  });

  factory AssetData.fromJson(Map<String, dynamic> json) {
    return AssetData(
      //**********************************************
      //* 추가: json에서 iv 필드를 가져와서 저장
      //**********************************************
      iv: json['iv'],
      cardData: CardData.fromJson(json['cardData']),
      demandDepositData: DemandDepositData.fromJson(json['demandDepositData']),
      loanData: LoanData.fromJson(json['loanData']),
      savingsData: SavingsData.fromJson(json['savingsData']),
      depositData: DepositData.fromJson(json['depositData']),
    );
  }
}