// presentation/widgets/bank_icon.dart
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/core/constants/icon_path.dart';

class BankIcon extends StatelessWidget {
  final String bankName;
  final double size;

  const BankIcon({
    Key? key, 
    required this.bankName, 
    this.size = 40,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final String iconPath = _getBankIconPath(bankName);
    
    if (_isPngPath(iconPath)) {
      return Image.asset(
        iconPath,
        width: size,
        height: size,
      );
    } else {
      return SvgPicture.asset(
        iconPath,
        width: size,
        height: size,
      );
    }
  }

  bool _isPngPath(String path) {
    return path.endsWith('.png');
  }

  String _getBankIconPath(String bankName) {
    switch (bankName.toLowerCase()) {
      case "한국은행":
      case "korea bank":
        return IconPath.koreaBank2;
      case "산업은행":
      case "kdb bank":
        return IconPath.kdbBank;
      case "기업은행":
      case "ibk bank":
        return IconPath.ibkBank;
      case "국민은행":
      case "kb bank":
        return IconPath.kbBank;
      case "농협은행":
      case "nh bank":
        return IconPath.nhBank;
      case "우리은행":
      case "woori bank":
        return IconPath.wooriBank;
      case "sc제일은행":
      case "standard chartered bank":
      case "sc bank":
        return IconPath.scBank;
      case "시티은행":
      case "citi bank":
        return IconPath.citiBank;
      case "대구은행":
      case "daegu bank":
        return IconPath.dgBank;
      case "광주은행":
      case "gwangju bank":
        return IconPath.gjBank;
      case "제주은행":
      case "jeju bank":
        return IconPath.jejuBank;
      case "전북은행":
      case "jeonbuk bank":
        return IconPath.jbBank;
      case "경남은행":
      case "gyeongnam bank":
        return IconPath.gnBank;
      case "새마을금고":
      case "mg":
        return IconPath.mgBank;
      case "keb하나은행":
      case "hana bank":
        return IconPath.hanaBank;
      case "신한은행":
      case "shinhan bank":
        return IconPath.shinhanBank;
      case "카카오뱅크":
      case "kakao bank":
        return IconPath.kakaoBank;
      case "싸피은행":
      case "ssafy bank":
      case "toss bank":
        return IconPath.ssafyBank2;
      case "-":
        return IconPath.gnBank; // 대출일 때 bankName 대신 "-"를 줌
      default:
        return IconPath.ibkBank; // 기본값으로 IBK 아이콘 사용
    }
  }
}