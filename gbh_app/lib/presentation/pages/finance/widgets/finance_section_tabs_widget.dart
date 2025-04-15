// 자산 페이지 내 섹션 탭을 위한 위젯
// 선택된 탭에 밑줄 표시 및 탭 클릭 시 해당 섹션으로 스크롤 기능 제공
import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';

class FinanceSectionTabs extends StatefulWidget {
  // 스크롤 컨트롤러 - 페이지의 스크롤 제어를 위해 전달받음
  final ScrollController scrollController;
  
  // 섹션별 스크롤 위치를 저장한 맵 - 각 섹션의 위치 정보
  final Map<String, GlobalKey> sectionKeys;
  
  const FinanceSectionTabs({
    super.key, 
    required this.scrollController,
    required this.sectionKeys,
  });

  @override
  State<FinanceSectionTabs> createState() => _FinanceSectionTabsState();
}

class _FinanceSectionTabsState extends State<FinanceSectionTabs> {
  // 현재 선택된 탭 인덱스
  int _selectedTabIndex = 0;
  
  // 섹션 제목 목록
  final List<String> _sectionTitles = ['입출금', '카드', '예적금', '대출'];
  
  // 탭 보이기/숨기기 상태
  bool _isVisible = true;
  
  // 스크롤 이전 위치 - 스크롤 방향 감지용
  double _previousScrollOffset = 0;

  @override
  void initState() {
    super.initState();
    
    // 스크롤 리스너 추가
    widget.scrollController.addListener(_scrollListener);
  }

  @override
  void dispose() {
    // 스크롤 리스너 제거
    widget.scrollController.removeListener(_scrollListener);
    super.dispose();
  }

  // 스크롤 이벤트 처리 리스너
  void _scrollListener() {
    // 1. 스크롤 방향 감지
    final currentOffset = widget.scrollController.offset;
    final isScrollingDown = currentOffset > _previousScrollOffset;
    
    // 2. 스크롤 방향에 따라 탭 보이기/숨기기
    if (isScrollingDown && _isVisible && currentOffset > 50) {
      // 아래로 스크롤하고 현재 보이는 상태라면 숨김
      setState(() => _isVisible = false);
    } else if (!isScrollingDown && !_isVisible) {
      // 위로 스크롤하고 현재 숨겨진 상태라면 보임
      setState(() => _isVisible = true);
    }
    
    // 3. 현재 스크롤 위치 기록
    _previousScrollOffset = currentOffset;
    
    // 4. 현재 화면에 보이는 섹션에 따라 탭 선택 상태 업데이트
    _updateSelectedTabBasedOnScroll();
  }
  
  // 현재 스크롤 위치에 따라 선택된 탭 업데이트
  void _updateSelectedTabBasedOnScroll() {
    // 현재 스크롤 위치 + 화면 중간 지점
    final scrollPosition = widget.scrollController.offset + (MediaQuery.of(context).size.height / 2);
    
    // 각 섹션의 위치를 확인하여 현재 위치에 맞는 탭 선택
    for (int i = 0; i < _sectionTitles.length; i++) {
      final key = widget.sectionKeys[_sectionTitles[i]];
      if (key?.currentContext != null) {
        final sectionBox = key!.currentContext!.findRenderObject() as RenderBox;
        final sectionPosition = sectionBox.localToGlobal(Offset.zero).dy;
        
        // 해당 섹션이 화면 중앙에 가까우면 탭 선택
        if (sectionPosition <= scrollPosition) {
          if (_selectedTabIndex != i) {
            setState(() => _selectedTabIndex = i);
          }
          // 더 아래 섹션도 확인하기 위해 break 하지 않음
        }
      }
    }
  }

  // 특정 섹션으로 스크롤
  void _scrollToSection(int index) {
    final sectionKey = widget.sectionKeys[_sectionTitles[index]];
    if (sectionKey?.currentContext != null) {
      // 해당 섹션 위치로 스크롤 애니메이션 적용
      Scrollable.ensureVisible(
        sectionKey!.currentContext!,
        duration: const Duration(milliseconds: 500),
        curve: Curves.easeInOut,
        alignment: 0.0, // 화면 최상단에 위치하도록
      );
      
      // 선택된 탭 업데이트
      setState(() => _selectedTabIndex = index);
    }
  }

  @override
  Widget build(BuildContext context) {
    // 탭 보이기/숨기기 애니메이션
    return AnimatedOpacity(
      opacity: _isVisible ? 1.0 : 0.0,
      duration: const Duration(milliseconds: 200),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        height: _isVisible ? 50 : 0,
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: Row(
            children: List.generate(
              _sectionTitles.length,
              (index) => _buildTab(index),
            ),
          ),
        ),
      ),
    );
  }

  // 개별 탭 위젯 생성
  Widget _buildTab(int index) {
    final isSelected = index == _selectedTabIndex;
    
    return GestureDetector(
      onTap: () => _scrollToSection(index),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Text(
              _sectionTitles[index],
              style: TextStyle(
                fontSize: 16,
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
                color: isSelected ? AppColors.textPrimary : AppColors.blackLight,
              ),
            ),
          ),
          // 고정 길이의 밑줄 (텍스트와 별개로 크기 지정)
          Container(
            width: 20, // 원하는 밑줄 길이로 조정
            height: 2,
            color: isSelected ? AppColors.textPrimary : Colors.transparent,
          ),
        ],
      ),
    );
  }
}