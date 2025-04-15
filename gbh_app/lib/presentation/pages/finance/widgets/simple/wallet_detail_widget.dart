import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:go_router/go_router.dart';
import 'package:lottie/lottie.dart';
import 'package:marshmellow/core/constants/icon_path.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/core/theme/bank_colors.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/finance/bank_icon.dart';
import 'package:marshmellow/router/routes/finance_routes.dart';

class WalletDetailWidget extends StatefulWidget {
  final String? walletType;
  final Map<String, dynamic>? walletData;
  final VoidCallback onBackPressed;

  const WalletDetailWidget({
    Key? key,
    required this.walletType,
    required this.walletData,
    required this.onBackPressed,
  }) : super(key: key);

  @override
  State<WalletDetailWidget> createState() => _WalletDetailWidgetState();
}

class _WalletDetailWidgetState extends State<WalletDetailWidget>
    with SingleTickerProviderStateMixin {
  late PageController _pageController;
  int _currentIndex = 0;
  late AnimationController _animationController;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _pageController = PageController(viewportFraction: 0.85);

    // 애니메이션 컨트롤러 설정
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );

    _animation =
        Tween<double>(begin: 0.0, end: 1.0).animate(_animationController);

    // 초기 애니메이션 실행
    _animationController.forward();
  }

  @override
  void dispose() {
    _pageController.dispose();
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 선택된 정보가 없으면 기본 메시지 표시
    if (widget.walletType == null || widget.walletData == null) {
      return Container(
        margin: const EdgeInsets.all(20),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
                color: Colors.black.withOpacity(0.1),
                blurRadius: 10,
                offset: const Offset(0, 4))
          ],
        ),
        child: const Text('지갑 정보를 불러올 수 없습니다.'),
      );
    }

    // 자산 추가하기는 별도 처리
    if (widget.walletType == '자산 추가하기') {
      return Container(
        margin: const EdgeInsets.all(20),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
                color: Colors.black.withOpacity(0.1),
                blurRadius: 10,
                offset: const Offset(0, 4))
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('새로운 자산을 추가해보세요', style: TextStyle(fontSize: 18)),
            const SizedBox(height: 36),
            Button(
              width: 200,
              onPressed: widget.onBackPressed,
              text: '뒤로',
            ),
          ],
        ),
      );
    }

    final List accountList = widget.walletData!['list'] as List;
    if (accountList.isEmpty) {
      return Container(
        margin: const EdgeInsets.all(20),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
                color: Colors.black.withOpacity(0.1),
                blurRadius: 10,
                offset: const Offset(0, 4))
          ],
        ),
        child: Column(
          children: [
            Text('${widget.walletType} 정보가 없습니다',
                style: TextStyle(fontSize: 18)),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: widget.onBackPressed,
              style: ElevatedButton.styleFrom(
                minimumSize: const Size(double.infinity, 48),
              ),
              child: const Text('뒤로'),
            ),
          ],
        ),
      );
    }

    // 구현에 필요한 제스처 감지기를 포함한 컨테이너
    return GestureDetector(
      onVerticalDragEnd: (details) {
        if (details.velocity.pixelsPerSecond.dy > 1000) {
          // 속도 임계값을 높임
          // 아래로 빠르게 스와이프하면 뒤로가기
          widget.onBackPressed();
        }
      },
      child: Container(
        // margin: const EdgeInsets.all(16),
        width: MediaQuery.of(context).size.width * 0.9,
        height: MediaQuery.of(context).size.height * 0.6,
        // color: Colors.amber,
        child: Column(
          // crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 계좌 카드 부분을 Stack으로 감싸서 고정 지갑과 스와이핑 카드 분리
            //             Padding(
            //   padding: const EdgeInsets.only(left: 16, right: 16, bottom: 8),
            //   child: Text(
            //     '${widget.walletType} 자산',
            //     style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            //   ),
            // ),
            // Text('@@ee@@${accountList[0].toString()}'), // Map<String, dynamic> Instance of 'DemandDepositItem'
            // Text('@@ee@@${accountList[0].encodedAccountBalance}'),
            // Text('@@ee@@${accountList[0].demandDepositData.totalAmount}'),
            Expanded(
              child: Stack(
                alignment: Alignment.center,
                children: [
                  // // 지갑 모양 뒷부분 (항상 같은 위치)
                  Positioned(
                      bottom: 0,
                      child: Container(
                        width: MediaQuery.of(context).size.width * 0.9,
                        height: 300,
                        decoration: BoxDecoration(
                            color: const Color.fromARGB(255, 46, 45, 45),
                            borderRadius:
                                BorderRadius.all(Radius.circular(30))),
                      )),

                  // 스와이핑되는 카드들만 PageView로 처리
                  Positioned.fill(
                    child: PageView.builder(
                      controller: _pageController,
                      scrollDirection: Axis.vertical,
                      itemCount: accountList.length,
                      onPageChanged: (index) {
                        setState(() {
                          _currentIndex = index;
                        });
                      },
                      itemBuilder: (context, index) {
                        return _buildCardContent(context, accountList, index);
                      },
                    ),
                  ),

                  // 지갑 모양 앞아래부분 (항상 같은 위치)
                  Positioned(
                      bottom: 0,
                      child: Container(
                        width: MediaQuery.of(context).size.width * 0.9,
                        height: 190,
                        decoration: BoxDecoration(
                            color: AppColors.backgroundBlack,
                            borderRadius: BorderRadius.only(
                                bottomLeft: Radius.circular(30),
                                bottomRight: Radius.circular(30))),
                      )),
                  // 지갑 모양 앞위 부분 (항상 같은 위치)
                  Positioned(
                    bottom: 160,
                    child: _buildWalletShape(AppColors.backgroundBlack,
                        MediaQuery.of(context).size.width),
                  ),

                  Positioned(
                    bottom: 55,
                    child: SvgPicture.asset(
                      IconPath.caretdoubledown,
                      colorFilter: ColorFilter.mode(
                          AppColors.background, BlendMode.srcIn),
                      height: 20,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // 카드 컨텐츠만 생성 (지갑 모양 없이)
  Widget _buildCardContent(
      BuildContext context, List accountList, int mainIndex) {
    final item = accountList[mainIndex];
    // print('@@@@@here@ : ${item.accountName}');
    // print('@@@@@her2e@ : ${item.accountBalance}');

    // 계좌 정보 추출
    String bankName = '';
    String accountNo = '';
    String accountName = '';
    dynamic balance;

    // 지갑 유형에 따라 다른 필드 사용
    switch (widget.walletType) {
      case '입출금':
        bankName = item.bankName;
        accountNo = item.accountNo;
        accountName = item.accountName;
        balance = item.encodedAccountBalance;
        break;
      case '예금':
        bankName = item.bankName;
        accountNo = item.accountNo;
        accountName = item.accountName;
        balance = item.encodeDepositBalance;
        break;
      case '적금':
        bankName = item.bankName;
        accountNo = item.accountNo;
        accountName = item.accountName;
        balance = item.encodedTotalBalance;
        break;
      case '카드':
        bankName = item.cardIssuerName;
        accountNo = item.cardNo;
        accountName = item.cardName;
        balance =
            item.cardBalance != null ? int.tryParse(item.cardBalance!) ?? 0 : 0;
        break;
      case '대출':
        bankName = ""; // 대출은 bankName이 없음
        accountNo = item.accountNo;
        accountName = item.accountName;
        balance = item.encodeLoanBalance;
        break;
      default:
        bankName = '알 수 없음';
        accountName = '정보 없음';
        balance = 0;
    }

    // 계좌 스택 생성 (메인 카드와 하위 카드들)
    return GestureDetector(
      onTap: () =>
          _onItemTap(context, accountNo, bankName, accountName, balance),
      child: Stack(
        alignment: Alignment.center,
        children: [
          // 메인 카드
          AnimatedBuilder(
            animation: _animation,
            builder: (context, child) {
              return Transform.translate(
                offset: Offset(0, -120),
                child: _buildMainCard(
                  bankName: bankName,
                  accountName: accountName,
                  accountNo: accountNo,
                  balance: balance,
                  mainIndex: mainIndex,
                  length: accountList.length.toString(),
                ),
              );
            },
          ),
          // Positioned(
          //   bottom: 245, // 위치 조정 필요할 수 있음
          //   right: 50,
          //   child: Row(
          //     crossAxisAlignment: CrossAxisAlignment.end,
          //     children: [
          //       Text(
          //         "${mainIndex + 1}",
          //         style: TextStyle(
          //           color: const Color.fromARGB(255, 44, 44, 44),
          //           fontWeight: FontWeight.w400,
          //           fontSize: 26,
          //         ),
          //       ),
          //       Text('/', style: AppTextStyles.appBar,),
          //       Text(accountList.length.toString(), style: AppTextStyles.appBar,)
          //     ],
          //   ),
          // ),
        ],
      ),
    );
  }

  // 메인 카드 위젯
  Widget _buildMainCard({
    required String bankName,
    required String accountName,
    required String accountNo,
    required int mainIndex,
    required String length,
    required dynamic balance,
  }) {
    return Container(
      width: MediaQuery.of(context).size.width * 0.8,
      height: 200,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: BankColors.getColorByBankName(bankName),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    // 은행 아이콘
                    IconButton(
                      icon: BankIcon(bankName: bankName, size: 40),
                      onPressed: () {},
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        accountName,
                        style: AppTextStyles.appBar.copyWith(
                            color: const Color.fromARGB(255, 31, 30, 30)),
                        overflow: TextOverflow.clip,
                      ),
                    ),
                    if (widget.walletType == '카드')
                      Icon(Icons.copy_outlined, size: 16),
                  ],
                ),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text(accountNo,
                      style: AppTextStyles.bodyMediumLight.copyWith(
                          color: const Color.fromARGB(255, 91, 91, 91))),
                ),
                Spacer(),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text('${_formatCurrency(balance)}원',
                      style: AppTextStyles.moneyBodyLarge.copyWith(
                          color: const Color.fromARGB(255, 25, 25, 25))

                      // style: TextStyle(
                      //   fontSize: 24,
                      //   fontWeight: FontWeight.bold,
                      //   color: widget.walletType == '대출' ? Colors.red : Colors.black,
                      // ),
                      ),
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                child: Center(
                  child: Transform.scale(
                    scale: 5,
                    child: Lottie.asset(
                      'assets/images/loading/scrollup.json',
                      width: 30,
                      height: 60,
                      fit: BoxFit.contain,
                    ),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 20),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      "${mainIndex + 1}",
                      style: TextStyle(
                        color: const Color.fromARGB(255, 44, 44, 44),
                        fontWeight: FontWeight.w400,
                        fontSize: 26,
                      ),
                    ),
                    Text(
                      '/',
                      style: AppTextStyles.appBar,
                    ),
                    Text(
                      length,
                      style: AppTextStyles.appBar,
                    )
                  ],
                ),
              ),
            ],
          )
        ],
      ),
    );
  }

  // 지갑 모양 위젯
  Widget _buildWalletShape(Color color, double width) {
    return ClipPath(
      clipper: BoxClipper(),
      child: Container(
        width: width * 0.65,
        height: width * 0.25,
        color: color,
      ),
    );
  }

  // 계좌 애니메이션 메서드
  void _animateCards() {
    _animationController.reset();
    _animationController.forward();
  }

  // 통화 형식 변환 헬퍼 메서드
  String _formatCurrency(dynamic amount) {
    if (amount == null) return '0';

    // 문자열로 변환하고 쉼표 추가
    String amountStr = amount.toString().replaceAll(RegExp(r'[^0-9]'), '');
    if (amountStr.isEmpty) return '0';

    // 3자리마다 쉼표 추가
    final regex = RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))');
    return amountStr.replaceAllMapped(regex, (Match match) => '${match[1]},');
  }

  // 항목 클릭 처리 메서드
  void _onItemTap(BuildContext context, String accountNo, String bankName,
      String accountName, dynamic balance) {
    print("항목 클릭: 유형=${widget.walletType}, 계좌번호=$accountNo");

    // noMoneyMan 값은 예시로 false로 설정. 실제 데이터에 맞게 조정 필요
    bool noMoneyMan = false;

    // balance를 반드시 정수로 변환
    int balanceInt = 0;

    // balance가 문자열이든 숫자든 안전하게 int로 변환
    if (balance != null) {
      if (balance is int) {
        balanceInt = balance;
      } else if (balance is String) {
        // 쉼표(,) 제거 후 정수로 변환 시도
        balanceInt =
            int.tryParse(balance.replaceAll(RegExp(r'[^\d]'), '')) ?? 0;
      } else {
        // 다른 타입일 경우 toString 후 정수 변환 시도
        balanceInt =
            int.tryParse(balance.toString().replaceAll(RegExp(r'[^\d]'), '')) ??
                0;
      }
    }

    // 자산 유형에 따라 다른 경로로 이동
    switch (widget.walletType) {
      case '입출금':
        context.push(
          FinanceRoutes.getDemandDetailPath(accountNo),
          extra: {
            'bankName': bankName,
            'accountName': accountName,
            'accountNo': accountNo,
            'balance': balanceInt, // int로 변환된 balance 사용
            'noMoneyMan': noMoneyMan,
          },
        );
        break;
      case '예금':
        context.push(
          FinanceRoutes.getDepositDetailPath(accountNo),
          extra: {
            'bankName': bankName,
            'accountName': accountName,
            'accountNo': accountNo,
            'balance': balanceInt, // int로 변환된 balance 사용
            'noMoneyMan': noMoneyMan,
          },
        );
        break;
      case '적금':
        context.push(
          FinanceRoutes.getSavingDetailPath(accountNo),
          extra: {
            'bankName': bankName,
            'accountName': accountName,
            'accountNo': accountNo,
            'balance': balanceInt, // int로 변환된 balance 사용
            'noMoneyMan': noMoneyMan,
          },
        );
        break;
      case '카드':
        context.push(
          FinanceRoutes.getCardDetailPath(accountNo),
          extra: {
            'bankName': bankName,
            'cardName': accountName,
            'cardNo': accountNo,
            'cvc': '123', // CVC 값은 실제 데이터에 맞게 조정 필요
            'balance': balanceInt, // int로 변환된 balance 사용
          },
        );
        break;
      case '대출':
        print("대출 path이동시도 ");
        context.push(
          FinanceRoutes.getLoanDetailPath(accountNo),
          extra: {
            'bankName': bankName,
            'accountName': accountName,
            'accountNo': accountNo,
            'balance': balanceInt, // int로 변환된 balance 사용
            'noMoneyMan': noMoneyMan,
          },
        );
        break;
      default:
        // 기본 처리 (필요시)
        break;
    }
  }
}

// 박스 앞쪽 상단 모양 클리퍼
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
