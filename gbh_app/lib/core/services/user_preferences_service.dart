import 'package:shared_preferences/shared_preferences.dart';

class UserPreferencesService {
  static const String _hasSeenBudgetFlow = 'has_seen_budget_flow';
  static const String _lastSalaryFlowDate = 'last_salary_flow_date';
  static String? _cachedLastFlowDate; // 성능 개선을 위한 메모리 캐싱

  // 사용자가 예산 축하 및 생성 플로우를 본 적 있는지 확인
  static Future<bool> hasSeenBudgetFlow() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_hasSeenBudgetFlow) ?? false;
  }

  // 사용자가 예산 플로우를 봤다고 표시
  static Future<void> markBudgetFlowAsSeen() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_hasSeenBudgetFlow, true);

    // 현재 날짜 저장(YYYY-MM형식)
    final now = DateTime.now();
    final monthKey = '${now.year}-${now.month.toString().padLeft(2, '0')}';
    await prefs.setString(_lastSalaryFlowDate, monthKey);

    // 캐시 업데이트
    _cachedLastFlowDate = monthKey;
  }

  // 이번 달에 이미 월급 플로우를 봤는지 확인
  static Future<bool> hasSeenSalaryFlowThisMonth() async {
    // 캐시된 값이 있으면 바로 사용
    if (_cachedLastFlowDate != null) {
      final now = DateTime.now();
      final currentMonthKey = '${now.year}-${now.month.toString().padLeft(2, '0')}';
      return _cachedLastFlowDate == currentMonthKey;
    }

    final prefs = await SharedPreferences.getInstance();
    final lastFlowDate = prefs.getString(_lastSalaryFlowDate);
    _cachedLastFlowDate = lastFlowDate; // 캐시에 저장

    if (lastFlowDate == null) return false;

    // 현재 월 조회
    final now = DateTime.now();
    final currentMonthKey = '${now.year}-${now.month.toString().padLeft(2, '0')}';

    // 이번달에 이미 봤으면 true 반환
    return lastFlowDate == currentMonthKey;
  }
}