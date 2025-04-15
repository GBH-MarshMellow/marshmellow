// presentation/pages/finance/widgets/sliding_assets_widget.dart
import 'package:flutter/material.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/simple/real_wallet_widget.dart';
import 'package:marshmellow/presentation/pages/finance/widgets/simple/wallet_detail_widget.dart';


class SlidingAssetsWidget extends StatefulWidget {
  final dynamic data; // 데이터 타입을 적절히 조정하세요

  const SlidingAssetsWidget({
    Key? key,
    required this.data,
  }) : super(key: key);

  @override
  State<SlidingAssetsWidget> createState() => _SlidingAssetsWidgetState();
}

class _SlidingAssetsWidgetState extends State<SlidingAssetsWidget> 
    with SingleTickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<Offset> _leftSlideAnimation;
  late Animation<Offset> _rightSlideAnimation;
  late Animation<Offset> _slideUpAnimation;
  late Animation<double> _fadeAnimation;
  bool _isSliding = false;
  bool _showNewWidget = false;
  String? _selectedWalletType; // 선택된 지갑 유형
  Map<String, dynamic>? _selectedWalletData; // 선택된 지갑 데이터

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800), // 애니메이션 시간 증가
    );

    // 왼쪽 슬라이드 애니메이션
    _leftSlideAnimation = Tween<Offset>(
      begin: Offset.zero,
      end: const Offset(-1.5, 0), // 왼쪽으로 슬라이드
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: const Interval(0.0, 0.6, curve: Curves.easeInOut), // 처음 60% 동안만 실행
    ));

    // 오른쪽 슬라이드 애니메이션
    _rightSlideAnimation = Tween<Offset>(
      begin: Offset.zero,
      end: const Offset(1.5, 0), // 오른쪽으로 슬라이드
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: const Interval(0.0, 0.6, curve: Curves.easeInOut), // 처음 60% 동안만 실행
    ));

    // 아래에서 위로 슬라이드 애니메이션
    _slideUpAnimation = Tween<Offset>(
      begin: const Offset(0, 1.0), // 화면 아래에서 시작
      end: Offset.zero, // 원래 위치로 이동
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: const Interval(0.6, 1.0, curve: Curves.easeOutQuad), // 후반 40%에서 실행
    ));

    // 페이드인 애니메이션
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: const Interval(0.6, 1.0, curve: Curves.easeIn), // 후반 40%에서 실행
    ));

    _animationController.addStatusListener((status) {
      if (status == AnimationStatus.completed) {
        setState(() {
          _isSliding = true;
          _showNewWidget = true; // 애니메이션 완료 후 새 위젯 표시
        });
      } else if (status == AnimationStatus.dismissed) {
        setState(() {
          _isSliding = false;
          _showNewWidget = false; // 애니메이션 되돌릴 때 새 위젯 숨김
        });
      }
    });
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        // 원래 Row 위젯 부분 - 애니메이션 값이 0.6 미만일 때 보임
        AnimatedOpacity(
          opacity: _animationController.value < 0.6 ? 1.0 : 0.0,
          duration: const Duration(milliseconds: 200),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // 왼쪽 열
              Expanded(
                child: SlideTransition(
                  position: _leftSlideAnimation,
                  child: Padding(
                    padding: const EdgeInsets.only(right: 9),
                    child: Column(
                      children: [
                        // 입출금 지갑
                        GestureDetector(
                          onTap: () {
                            setState(() {
                              _selectedWalletType = '입출금';
                              _selectedWalletData = {
                                'list': widget.data.data.demandDepositData.demandDepositList,
                                'totalAmount': int.tryParse(widget.data.data.demandDepositData.totalAmount) ?? 0,
                              };
                            });
                            if (!_isSliding) {
                              _animationController.forward();
                            }
                          },
                          child: RealWallet(
                            demandDepositList: widget.data.data.demandDepositData.demandDepositList.map((item) => 
                              DemandDepositItem(
                                bankName: item.bankName,
                                accountName: item.accountName,
                                accountNo: item.accountNo,
                                accountBalance: item.accountBalance,
                              )
                            ).toList().cast<DemandDepositItem>(),
                            totalAmount: int.tryParse(widget.data.data.demandDepositData.totalAmount) ?? 0,
                            type: '입출금',
                          ),
                        ),
                        SizedBox(height: 24),
                        
                        // 예금 지갑
                        GestureDetector(
                          onTap: () {
                            setState(() {
                              _selectedWalletType = '예금';
                              _selectedWalletData = {
                                'list': widget.data.data.depositData.depositList,
                                'totalAmount': int.tryParse(widget.data.data.depositData.totalAmount) ?? 0,
                              };
                            });
                            if (!_isSliding) {
                              _animationController.forward();
                            }
                          },
                          child: RealWallet(
                            demandDepositList: widget.data.data.depositData.depositList.map((item) => 
                              DemandDepositItem(
                                bankName: item.bankName,
                                accountName: item.accountName,
                                accountNo: item.accountNo,
                                accountBalance: item.depositBalance,
                              )
                            ).toList().cast<DemandDepositItem>(),
                            totalAmount: int.tryParse(widget.data.data.depositData.totalAmount) ?? 0,
                            type: '예금',
                          ),
                        ),
                        SizedBox(height: 24),
                        
                        // 대출 지갑
                        GestureDetector(
                          onTap: () {
                            setState(() {
                              _selectedWalletType = '대출';
                              _selectedWalletData = {
                                'list': widget.data.data.loanData.loanList,
                                'totalAmount': int.tryParse(widget.data.data.loanData.totalAmount) ?? 0,
                              };
                            });
                            if (!_isSliding) {
                              _animationController.forward();
                            }
                          },
                          child: RealWallet(
                            demandDepositList: widget.data.data.loanData.loanList.map((item) => 
                              DemandDepositItem(
                                bankName: item.accountName,
                                accountName: item.accountName,
                                accountNo: item.accountNo,
                                accountBalance: item.loanBalance,
                              )
                            ).toList().cast<DemandDepositItem>(),
                            totalAmount: int.tryParse(widget.data.data.loanData.totalAmount) ?? 0,
                            type: '대출',
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              
              // 오른쪽 열
              Expanded(
                child: SlideTransition(
                  position: _rightSlideAnimation,
                  child: Padding(
                    padding: const EdgeInsets.only(left: 9),
                    child: Column(
                      children: [
                        // 카드 지갑
                        GestureDetector(
                          onTap: () {
                            setState(() {
                              _selectedWalletType = '카드';
                              _selectedWalletData = {
                                'list': widget.data.data.cardData.cardList,
                                'totalAmount': int.tryParse(widget.data.data.cardData.totalAmount) ?? 0,
                              };
                            });
                            if (!_isSliding) {
                              _animationController.forward();
                            }
                          },
                          child: RealWallet(
                            demandDepositList: widget.data.data.cardData.cardList.map((item) => 
                              DemandDepositItem(
                                bankName: item.cardName,
                                accountName: item.cardIssuerName,
                                accountNo: item.cardNo,
                                accountBalance: item.cardBalance != null ? int.tryParse(item.cardBalance!) ?? 0 : 0,
                              )
                            ).toList().cast<DemandDepositItem>(),
                            totalAmount: int.tryParse(widget.data.data.cardData.totalAmount) ?? 0,
                            type: '카드',
                          ),
                        ),
                        SizedBox(height: 24),
                        
                        // 적금 지갑
                        GestureDetector(
                          onTap: () {
                            setState(() {
                              _selectedWalletType = '적금';
                              _selectedWalletData = {
                                'list': widget.data.data.savingsData.savingsList,
                                'totalAmount': int.tryParse(widget.data.data.savingsData.totalAmount ) ?? 10,
                              };
                            });
                            if (!_isSliding) {
                              _animationController.forward();
                            }
                          },
                          child: RealWallet(
                            demandDepositList: widget.data.data.savingsData.savingsList.map((item) => 
                              DemandDepositItem(
                                bankName: item.bankName,
                                accountName: item.accountName,
                                accountNo: item.accountNo,
                                accountBalance: item.totalBalance,
                              )
                            ).toList().cast<DemandDepositItem>(),
                            totalAmount: int.tryParse(widget.data.data.savingsData.totalAmount) ?? 0,
                            type: '적금',
                          ),
                        ),
                        SizedBox(height: 24),
                        
                        // 자산 추가하기
                        GestureDetector(
                          onTap: () {
                            setState(() {
                              _selectedWalletType = '자산 추가하기';
                              _selectedWalletData = {
                                'list': [],
                                'totalAmount': -1,
                              };
                            });
                            if (!_isSliding) {
                              _animationController.forward();
                            }
                          },
                          child: RealWallet(
                            demandDepositList: [],
                            totalAmount: -1,
                            type: '자산 추가하기',
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
        
        // 새 위젯 - 애니메이션 값이 0.6 이상일 때 보임
        AnimatedBuilder(
          animation: _animationController,
          builder: (context, child) {
            return _animationController.value >= 0.6
                ? SlideTransition(
                    position: _slideUpAnimation,
                    child: FadeTransition(
                      opacity: _fadeAnimation,
                      child: SizedBox(
                        width: double.infinity,
                        child: WalletDetailWidget(
                          walletType: _selectedWalletType,
                          walletData: _selectedWalletData,
                          onBackPressed: () => _animationController.reverse(),
                        ),
                      ),
                    ),
                  )
                : SizedBox.shrink();
          },
        )
      ],
    );
  }

}