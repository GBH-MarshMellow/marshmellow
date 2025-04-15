// lib/presentation/pages/budget/widgets/budget_salary/budget_creation_page.dart

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/data/models/budget/budget_model.dart';
import 'package:marshmellow/data/models/budget/budget_type_model.dart';
import 'package:marshmellow/presentation/pages/budget/widgets/budget_bubble_chart.dart';
import 'package:marshmellow/presentation/viewmodels/budget/budget_type_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/budget/budget_viewmodel.dart';
import 'package:marshmellow/presentation/viewmodels/my/user_info_viewmodel.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/widgets/button/button.dart';
import 'package:marshmellow/presentation/widgets/loading/custom_loading_indicator.dart';
import 'package:marshmellow/router/routes/auth_routes.dart';
import 'package:marshmellow/router/routes/budget_routes.dart';

class BudgetCreationPage extends ConsumerStatefulWidget {
  final String selectedType;
  // final int salary;

  const BudgetCreationPage({
    Key? key,
    required this.selectedType,
  }) : super(key: key);

  @override
  ConsumerState<BudgetCreationPage> createState() => _BudgetCreationPageState();
}

class _BudgetCreationPageState extends ConsumerState<BudgetCreationPage> {
  bool _isLoading = true;
  bool _isEditing = false;
  String? _errorMessage;
  BudgetModel? _createdBudget;
  
  // 수정할 카테고리 정보
  int? _selectedCategoryPk;
  TextEditingController _amountController = TextEditingController();
  bool _isEditingAmount = false;
  FocusNode _amountFocusNode = FocusNode();
  
  @override
  void initState() {
    super.initState();
    // 페이지 로드 시 예산 생성 API 호출
    Future.microtask(() {
      _createBudget();  
    });
  }
  
  @override
  void dispose() {
    _amountController.dispose();
    _amountFocusNode.dispose();
    super.dispose();
  }

  // 예산 생성 API 호출
  Future<void> _createBudget() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      // UserInfo에서 salary 정보 가져오기
      final userInfoState = ref.read(userInfoProvider);
      final salary = userInfoState.userDetail.salaryAmount;

      print('✨✨✨✨내월급 : $salary✨✨✨✨');

      if (salary == null) {
        setState(() {
          _isLoading = false;
          _errorMessage = '월급 정보를 찾을 수 없습니다';
        });
        return;
      }

      // 선택된 유형에 따른 지출 비율 가져오기
      final budgetTypeVM = ref.read(budgetTypeProvider.notifier);
      final selectedTypeData = budgetTypeVM.getSelectedTypeData();
      
      if (selectedTypeData == null) {
        setState(() {
          _isLoading = false;
          _errorMessage = '선택된 유형 데이터를 찾을 수 없습니다.';
        });
        return;
      }
      
      // 비율 데이터 변환
      final expenseData = selectedTypeData.toMap();
      
      // 예산 생성 API 호출
      await ref.read(budgetProvider.notifier).createBudget(
        salary: salary,
        fixedExpense: expenseData['고정지출'] ?? 0.0,
        foodExpense: expenseData['식비/외식'] ?? 0.0,
        transportationExpense: expenseData['교통/자동차'] ?? 0.0,
        marketExpense: expenseData['편의점/마트'] ?? 0.0,
        financialExpense: expenseData['금융'] ?? 0.0,
        leisureExpense: expenseData['여가비'] ?? 0.0,
        coffeeExpense: expenseData['커피/디저트'] ?? 0.0,
        shoppingExpense: expenseData['쇼핑'] ?? 0.0,
        emergencyExpense: expenseData['비상금'] ?? 0.0,
      );
      
      // 예산 생성 후 바로 전체 예산 목록 다시 불러오기
      await ref.read(budgetProvider.notifier).fetchBudgets();
      
      // 목록에서 첫 번째 예산(가장 최근 생성된 예산)을 가져옴
      final budgetState = ref.read(budgetProvider);
      if (budgetState.budgets.isNotEmpty) {
        _createdBudget = budgetState.budgets[0];
          print('✅ 생성된 예산 정보: ');
          print('  - 예산 ID: ${_createdBudget!.budgetPk}');
          print('  - 총액: ${_createdBudget!.budgetAmount}');
          print('  - 기간: ${_createdBudget!.startDate} ~ ${_createdBudget!.endDate}');
          print('  - 카테고리 수: ${_createdBudget!.budgetCategoryList.length}');
        setState(() {
          _isLoading = false;
        });
      } else {
        setState(() {
          _isLoading = false;
          _errorMessage = '생성된 예산 정보를 불러올 수 없습니다.';
        });
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = '예산 생성 중 오류가 발생했습니다: $e';
      });
    }
  }
  
  // 카테고리 예산 수정
  Future<void> _updateCategoryBudget() async {
    if (_selectedCategoryPk == null) return;

    // 입력된 금액 가져오기
    final amountText = _amountController.text.replaceAll(',', '');
    if (amountText.isEmpty) return;
    
    final amount = int.tryParse(amountText);
    if (amount == null) return;

    print('💰 카테고리 예산 업데이트:');
    print('  - 카테고리 ID: $_selectedCategoryPk');
    print('  - 변경 금액: $amount');
    
    setState(() {
      _isEditing = true;
    });
    
    try {
      // 카테고리 예산 업데이트
      await ref.read(budgetProvider.notifier).updateBudgetCategory(
        _selectedCategoryPk!,
        amount,
      );
      
      // 업데이트 후 예산 정보 새로고침
      final budgetState = ref.read(budgetProvider);
      if (budgetState.budgets.isNotEmpty) {
        _createdBudget = budgetState.budgets[0];
      }
      
      // 수정 모드 종료
      setState(() {
        _isEditing = false;
        _isEditingAmount = false;
        _selectedCategoryPk = null;
        _amountController.clear();
      });
      
    } catch (e) {
      setState(() {
        _isEditing = false;
        _errorMessage = '예산 수정 중 오류가 발생했습니다: $e';
      });
    }
  }
  
  // 카테고리 선택 및 편집 모드 진입
  void _selectCategory(int categoryPk, int initialAmount) {
    // 이미 선택된 카테고리를 다시 선택했을 경우 선택 해제
    if (_selectedCategoryPk == categoryPk && _isEditingAmount) {
      setState(() {
        _selectedCategoryPk = null;
        _isEditingAmount = false;
        _amountController.clear();
      });
      return;
    }

    setState(() {
      _selectedCategoryPk = categoryPk;
      _amountController.text = initialAmount.toString();
      _isEditingAmount = true;
    });
    
    // 키보드 포커스 주기
    Future.delayed(Duration(milliseconds: 100), () {
      _amountFocusNode.requestFocus();
    });

    // 선택된 카테고리가 화면에 보이도록 스크롤 조정
    // 이 부분은 필요에 따라 구현할 수 있습니다
  }
  
  // 금액 포맷팅 (천 단위 쉼표)
  String _formatAmount(int amount) {
    return amount.toString().replaceAllMapped(
      RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
      (Match m) => '${m[1]},'
    );
  }

  // 선택된 카테고리 이름 가져오기
  String _getSelectedCategoryName() {
    if (_selectedCategoryPk == null || _createdBudget == null) return '';
    
    final selectedCategory = _createdBudget!.budgetCategoryList.firstWhere(
      (category) => category.budgetCategoryPk == _selectedCategoryPk,
      orElse: () => BudgetCategoryModel(
        budgetCategoryPk: 0,
        budgetCategoryName: '',
        budgetCategoryPrice: 0,
      ),
    );
    
    return selectedCategory.budgetCategoryName;
  }

  // 선택된 카테고리 색상 가져오기
  Color _getSelectedCategoryColor() {
    if (_selectedCategoryPk == null || _createdBudget == null) return Colors.grey;
    
    final selectedCategory = _createdBudget!.budgetCategoryList.firstWhere(
      (category) => category.budgetCategoryPk == _selectedCategoryPk,
      orElse: () => BudgetCategoryModel(
        budgetCategoryPk: 0,
        budgetCategoryName: '',
        budgetCategoryPrice: 0,
      ),
    );
    
    // BudgetModel의 정적 메서드 사용
    return BudgetModel.getCategoryColor(selectedCategory.budgetCategoryName);
  }
  
  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        body: const Center(
          child: CustomLoadingIndicator(
            backgroundColor: AppColors.whiteLight,
            text: '예산 생성 중',
          ),
        ),
      );
    }

    if (_isEditing) {
      return Scaffold(
        body: const Center(
          child: CustomLoadingIndicator(
            backgroundColor: AppColors.whiteLight,
            text: '예산 수정 중',
          ),
        ),
      );
    }

    if (_errorMessage != null) {
      return Scaffold(
        appBar: CustomAppbar(title: '오류'),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(_errorMessage!),
              const SizedBox(height: 16),
              Button(
                text: '다시 시도',
                onPressed: _createBudget,
              ),
            ],
          ),
        ),
      );
    }

    if (_createdBudget == null) {
      return Scaffold(
        appBar: CustomAppbar(title: '오류'),
        body: const Center(
          child: Text('예산 정보를 불러올 수 없습니다.'),
        ),
      );
    }

    final categories = _createdBudget!.budgetCategoryList;
    
    // 예산 기간 가져오기
    final startDate = DateTime.parse(_createdBudget!.startDate);
    final endDate = DateTime.parse(_createdBudget!.endDate);
    final startMonth = startDate.month;
    final endMonth = endDate.month;
    
    // 타이틀 설정
    final title = '$startMonth월 예산 설정';
        
    // 금액 포맷팅
    final totalBudget = _formatAmount(_createdBudget!.budgetAmount);

    // 날짜 포맷팅
    String _formatDate(String dateString) {
      final date = DateTime.parse(dateString);
      return '${date.month}.${date.day}';
    }

    return Scaffold(
      backgroundColor: AppColors.whiteDark,
      appBar: CustomAppbar(
        title: title,
        backgroundColor: AppColors.whiteDark
      ),
      body: GestureDetector(
        onTap: () {
          // 빈 영역 터치 시 편집 모드 종료
          if (_isEditingAmount) {
            setState(() {
              _isEditingAmount = false;
              _selectedCategoryPk = null;
              _amountController.clear();
            });
          }
        },
        child: Stack(
          children: [
            // 스크롤 가능한 메인 컨텐츠
            SingleChildScrollView(
              child: Column(
                children: [
                  // 예산 정보 요약
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 22.0, vertical: 8.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        // 좌측 : 전체 예산
                        RichText(
                          text: TextSpan(
                            children: [
                              TextSpan(
                                text: totalBudget,
                                style: AppTextStyles.congratulation
                              ),
                              const TextSpan(
                                text: ' 원',
                                style: AppTextStyles.bodyMedium
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 8),

                  Container(
                    margin: const EdgeInsets.symmetric(vertical: 10),
                    child: Center(
                      child: Text(
                        '${_formatDate(_createdBudget!.startDate)} - ${_formatDate(_createdBudget!.endDate)}',
                        style: AppTextStyles.bodyLargeLight.copyWith(
                            fontWeight: FontWeight.w400
                          ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 10),
                  
                  // 메인 버블 차트 
                  Container(
                    height: 300, // 적절한 높이 지정
                    padding: const EdgeInsets.symmetric(horizontal: 22.0),
                    child: categories.isNotEmpty
                        ? BudgetBubblechart(categories: categories, enableNavigation: false)
                        : const Center(child: Text('등록된 예산이 없습니다')),
                  ),

                  const SizedBox(height: 25),

                  // 카테고리별 예산 목록
                  Padding(
                    padding: const EdgeInsets.all(22.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Center(
                          child: Text(
                          '카테고리',
                          style: AppTextStyles.bodyLargeLight.copyWith(
                            fontWeight: FontWeight.w400
                          ),
                          ),
                        ),
                        const SizedBox(height: 12),
                        ...categories.map((category) => _buildCategoryItem(category)),
                        const SizedBox(height: 10), 
                        Container(
                          padding: const EdgeInsets.all(16),
                          child: Button(
                            text: '완료',
                            onPressed: () {
                              // 편집 중이면 저장 먼저 처리
                              if (_isEditingAmount) {
                                _updateCategoryBudget();
                              }
                              // 위시 생성 페이지로 이동
                              context.go(SignupRoutes.getWishCreatePath());
                            },
                          ),
                        ),// 키보드 공간 확보
                      ],
                    ),
                  ),
                ],
              ),
            ),
            
            // 수정 중일 때 안내 메시지 표시 (선택적)
            if (_isEditingAmount)
              Positioned(
                left: 0,
                right: 0,
                bottom: 0,
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                  color: AppColors.whiteLight,
                  child: Text(
                    '${_getSelectedCategoryName()} 카테고리의 금액을 수정 중입니다',
                    style: AppTextStyles.bodySmall.copyWith(
                      fontWeight: FontWeight.w300,
                      color: AppColors.greyLight,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
  
  // 선택된 예산 유형 정보 위젯
  Widget _buildSelectedTypeInfo() {
    // BudgetTypeInfo 가져오기
    final typeInfo = BudgetTypeInfo.getTypeInfo(widget.selectedType);
    
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 22, vertical: 10),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: typeInfo.color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Container(
            width: 50,
            height: 50,
            decoration: BoxDecoration(
              color: typeInfo.color.withOpacity(0.2),
              shape: BoxShape.circle,
            ),
            child: Icon(
              _getIconForType(typeInfo.type),
              color: typeInfo.color,
              size: 30,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  typeInfo.selectionName,
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  typeInfo.description,
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[700],
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
  
  // 카테고리 아이템 위젯
  Widget _buildCategoryItem(BudgetCategoryModel category) {
    // 선택된 카테고리인지 확인
    final isSelected = _selectedCategoryPk == category.budgetCategoryPk;
    
    // 금액 포맷팅
    final formattedAmount = _formatAmount(category.budgetCategoryPrice);
    
    return InkWell(
      onTap: () {
        _selectCategory(category.budgetCategoryPk, category.budgetCategoryPrice);
      },
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
        decoration: BoxDecoration(
          border: Border(
            bottom: BorderSide(
              color: Colors.grey.withOpacity(0.3),
              width: 0.5,
            ),
          ),
          // color: isSelected ? AppColors.whiteLight : Colors.transparent,
        ),
        child: Row(
          children: [
            // 카테고리 색상 표시
            Container(
              width: 16,
              height: 16,
              decoration: BoxDecoration(
                color: category.color,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 12),
            // 카테고리 이름
            Expanded(
              child: Text(
                category.budgetCategoryName,
                style: AppTextStyles.bodyMediumLight.copyWith(fontWeight: FontWeight.w300),
              ),
            ),
            // 금액 표시 (수정 모드일 때는 TextField, 아닐 때는 Text)
            isSelected && _isEditingAmount
                ? Expanded(
                    child: TextField(
                      controller: _amountController,
                      focusNode: _amountFocusNode,
                      keyboardType: TextInputType.number,
                      textAlign: TextAlign.end,
                      decoration: InputDecoration(
                        border: InputBorder.none,
                        enabledBorder: InputBorder.none,
                        focusedBorder: InputBorder.none,
                        contentPadding: EdgeInsets.zero,
                        isDense: true,
                        suffixText: '원',
                      ),
                      style: AppTextStyles.bodyMedium.copyWith(
                        fontWeight: FontWeight.w300,
                        color: AppColors.textPrimary,
                      ),
                      inputFormatters: [
                        FilteringTextInputFormatter.digitsOnly,
                      ],
                      onSubmitted: (_) => _updateCategoryBudget(),
                    ),
                  )
                : Text(
                    '$formattedAmount 원',
                    style: AppTextStyles.bodyMedium.copyWith(
                      fontWeight: FontWeight.w300,
                    ),
                  ),
            // 편집 아이콘 (수정 모드일 때만 표시)
            if (isSelected && _isEditingAmount)
              IconButton(
                icon: const Icon(Icons.check, size: 20),
                padding: EdgeInsets.zero,
                constraints: BoxConstraints(),
                onPressed: _updateCategoryBudget,
              ),
            const SizedBox(width: 6),
          ],
        ),
      ),
    );
  }

  // 유형별 아이콘 반환
  IconData _getIconForType(String type) {
    switch (type) {
      case '식비/외식':
        return Icons.restaurant;
      case '교통/자동차':
        return Icons.directions_car;
      case '편의점/마트':
        return Icons.shopping_basket;
      case '금융':
        return Icons.account_balance;
      case '여가비':
        return Icons.movie;
      case '커피/디저트':
        return Icons.local_cafe;
      case '쇼핑':
        return Icons.shopping_bag;
      case '비상금':
        return Icons.savings;
      case '평균':
        return Icons.balance;
      default:
        return Icons.category;
    }
  }
}