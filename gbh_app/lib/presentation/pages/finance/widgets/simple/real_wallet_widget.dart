import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:marshmellow/presentation/widgets/finance/bank_icon.dart';
import 'package:marshmellow/presentation/widgets/finance/card_image_util.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart'; // 추가
import 'package:marshmellow/presentation/viewmodels/finance/finance_viewmodel.dart';
import 'package:marshmellow/core/theme/bank_colors.dart'; // 새로 추가한 import

// 계좌 정보 모델 (여기서만 써서)
// 클래스 이름만 변경하고 내부 필드는 그대로 유지
class DemandDepositItem {
  final String bankName;
  final String accountName;
  final String accountNo;
  final int accountBalance;
  
  DemandDepositItem({
    required this.bankName,
    required this.accountName,
    required this.accountNo,
    required this.accountBalance,
  });
}

// 앞서 만든 CardInfo 클래스
class CardInfo {
  final String cardName;
  final Color cardColor;
  final String? cardNumber;
  
  CardInfo({
    required this.cardName,
    required this.cardColor,
    this.cardNumber,
  });
}

// 계좌 정보를 CardInfo로 변환하는 함수 (DemandDepositItem으로 파라미터 타입 변경)
List<CardInfo> convertToCardInfoList(List<DemandDepositItem> accounts) {
  // 입출금 계좌가 없거나 1개만 있는 경우 처리
  if (accounts.isEmpty) {
    return [
      CardInfo(cardName: '계좌 없음', cardColor: Colors.white, cardNumber: null),
      CardInfo(cardName: '계좌 없음', cardColor: Colors.white, cardNumber: null),
    ];
  } else if (accounts.length == 1) {
    // 계좌가 1개인 경우, 하나는 실제 계좌, 하나는 빈 계좌
    return [
      CardInfo(
        cardName: accounts[0].bankName,
        cardColor: BankColors.getColorByBankName(accounts[0].bankName),
        cardNumber: accounts[0].accountNo.substring(max(0, accounts[0].accountNo.length - 4)),
      ),
      CardInfo(cardName: '계좌 없음', cardColor: Colors.white, cardNumber: null),
    ];
  } else {
    // 계좌가 2개 이상인 경우, 첫 2개만 사용
    return accounts.take(2).map((account) => CardInfo(
      cardName: account.bankName,
      cardColor: BankColors.getColorByBankName(account.bankName),
      cardNumber: account.accountNo.substring(max(0, account.accountNo.length - 4)),
    )).toList();
  }
}

// 최솟값 반환 함수
int max(int a, int b) {
  return a > b ? a : b;
}

// 월렛 위젯 구현
class DefaultWalletWidget extends ConsumerWidget {
  final List<CardInfo> cards;
  final String accountType;
  final int balance;

  
  const DefaultWalletWidget({
    Key? key,
    required this.cards,
    required this.accountType,
    required this.balance,
  }) : assert(cards.length == 2, '카드는 정확히 2개가 필요합니다'),
      super(key: key);
  
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isHidden = ref.watch(isFinanceHideProvider);
    return Container(
      width: MediaQuery.of(context).size.width * 0.4,
      height: MediaQuery.of(context).size.width * 0.4,
      decoration: BoxDecoration(
        color: AppColors.blackPrimary,
        borderRadius: BorderRadius.circular(24),
      ),
      child: Column(
        children: [
          // 카드 2장
          Container(
            child: Stack(
              alignment: Alignment.center,
              children: [
                _buildStackedCards(context),
              ],
            ),
          ),
          const SizedBox(height: 30.0),
          // 하단 검은색 영역 (계좌 정보)
          Expanded(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              width: double.infinity,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    accountType,
                    style: AppTextStyles.bodySmall.copyWith(color: AppColors.whiteLight)
                  ),
                  if (balance != -1)
                  Text(isHidden ? '금액보기' :
                    '${_formatCurrency(balance)}원',
                    style: isHidden ?  AppTextStyles.bodyMediumLight.copyWith(color: AppColors.divider , fontWeight: FontWeight.w400) :
                        AppTextStyles.bodySmall.copyWith(color: AppColors.whiteLight)
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildStackedCards(context) {
    double parentWidth = MediaQuery.of(context).size.width * 0.4;
    return Stack(
      alignment: Alignment.center,
      
      children: [
        // 첫 번째 카드 (뒤에 있는 카드)
        Transform.translate(
          offset: const Offset(0, 10),
          child: _buildCard(cards[1], isFirst: true, parentWidth: parentWidth),
        ),
        
        // 두 번째 카드 (앞에 있는 카드)
        Transform.translate(
          offset: const Offset(0, 40),
          child: _buildCard(cards[0], isFirst: false, parentWidth: parentWidth),
        ),
        Transform.translate(
          offset: const Offset(0, 55),
          child: CustomShapeBox(
            width: parentWidth * 0.65,
            height: parentWidth * 0.25,
            color: AppColors.blackPrimary
          ),
        ),
      ],
    );
  }
  
  Widget _buildCard(CardInfo card, {required bool isFirst, required double parentWidth}) {
    return Container(
      width: parentWidth * 0.85,
      height: parentWidth * 0.25,
      decoration: BoxDecoration(
        color: card.cardColor,
        borderRadius: BorderRadius.only(topLeft: Radius.circular(10.0),topRight: Radius.circular(10.0) ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.8),
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      padding: const EdgeInsets.fromLTRB(0.0, 0.0, 8.0, 0.0),
    child: card.cardName == "계좌 없음"
        ? Center(
            child: IconButton(
              icon: SvgPicture.asset(IconPath.plus,
              colorFilter: ColorFilter.mode(AppColors.blackLight, BlendMode.srcIn),),
              onPressed: () {
                // _onAccountItemTap(context);
              },
            ),
          )
        : accountType == "대출"
            ? Row(
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                IconButton(
                  icon: BankIcon(bankName: '싸피은행', size: 40),
                  onPressed: () {
                    // _onAccountItemTap(context);
                  },
                ),
                Expanded(
                  child: Container(
                      padding: const EdgeInsets.fromLTRB(0.0, 11.0, 0.0, 0.0),
                      alignment: Alignment.topCenter,
                      child: Text(
                        card.cardName,
                        style: AppTextStyles.bodySmall.copyWith(color: AppColors.blackPrimary),
                        overflow: TextOverflow.clip,
                        maxLines: 1,
                      ),
                    ),
                ),
              ],
            )
            : accountType == "카드" ? Row(
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  SizedBox(
                    width: 40,
                    child: Center(
                      child: CardImageUtil.getCardImageWidget(card.cardName, size: 30),
                    ),
                  ),
                  Expanded(  // 여기가 중요한 변경점입니다
                    child: Container(
                      alignment: Alignment.topCenter,
                      padding: const EdgeInsets.fromLTRB(0.0, 11.0, 0.0, 0.0),
                      child: Text(
                        card.cardName,
                        style: AppTextStyles.bodySmall.copyWith(color: AppColors.blackPrimary),
                        overflow: TextOverflow.clip,
                        maxLines: 1,
                      ),
                    ),
                  ),
                ],
              ) : Row(
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  IconButton(
                    icon: BankIcon(bankName: card.cardName, size: 40),
                    onPressed: () {
                      // _onAccountItemTap(context);
                    },
                  ),
                  Expanded(  // 여기가 중요한 변경점입니다
                    child: Container(
                      alignment: Alignment.topCenter,
                      padding: const EdgeInsets.fromLTRB(0.0, 11.0, 0.0, 0.0),
                      child: Text(
                        card.cardName,
                        style: AppTextStyles.bodySmall.copyWith(color: AppColors.blackPrimary),
                        overflow: TextOverflow.clip,
                        maxLines: 1,
                      ),
                    ),
                  ),
                ],
              )
    );
  }
  
  String _formatCurrency(int amount) {
    return amount.toString().replaceAllMapped(
      RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
      (Match m) => '${m[1]},',
    );
  }
}

// 기존 코드를 대체할 위젯 (DemandDepositItem으로 파라미터 타입 변경)
class RealWallet extends StatelessWidget {
  final List<DemandDepositItem> demandDepositList;
  final int totalAmount;
  final String type;
  
  const RealWallet({
    Key? key, 
    required this.demandDepositList, 
    required this.totalAmount,
    required this.type
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    // 계좌 정보를 CardInfo로 변환
    final List<CardInfo> cardInfoList = convertToCardInfoList(demandDepositList);
    
    // Wallet 위젯 반환
    return DefaultWalletWidget(
      cards: cardInfoList,
      accountType: type,
      balance: totalAmount,
    );
  }
} 

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class CustomShapeBox extends StatelessWidget {
  final double width;
  final double height;
  final Color color;
  const CustomShapeBox({
    Key? key,
    required this.width,
    required this.height,
    required this.color,
  }) : super(key: key);
  @override
  Widget build(BuildContext context) {
    return ClipPath(
      clipper: BoxClipper(),
      child: Container(
        width: width,
        height: height,
        color: color,
      ),
    );
  }
}
class BoxClipper extends CustomClipper<Path> {
  @override
  Path getClip(Size size) {
    final path = Path();
    
    // 곡선 깊이 조절 (값을 조절하여 곡선의 정도를 바꿀 수 있습니다)
    final curveHeight = size.height * 0.35;
    
    // 시작점 (왼쪽 하단)
    path.moveTo(0, size.height);
    
    // 왼쪽 상단 곡선
    path.quadraticBezierTo(0, curveHeight, curveHeight, curveHeight);
    
    // 상단 직선
    path.lineTo(size.width - curveHeight, curveHeight);
    
    // 오른쪽 상단 곡선
    path.quadraticBezierTo(size.width, curveHeight, size.width, size.height);
    
    // 완성
    path.close();
    
    return path;
  }
  @override
  bool shouldReclip(covariant CustomClipper<Path> oldClipper) {
    return false;
  }
}