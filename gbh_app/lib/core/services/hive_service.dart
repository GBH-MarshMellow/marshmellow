import 'package:shared_preferences/shared_preferences.dart';

class SearchHistoryService {
  static const String _key = 'search_history';

  // 검색어 저장
  static Future<void> saveSearch(String query) async {
    final prefs = await SharedPreferences.getInstance();

    // 기존 검색 히스토리 불러오기
    List<String> history = prefs.getStringList(_key) ?? [];

    // 중복 제거 및 최신 검색어 추가
    history.remove(query);
    history.insert(0, query);

    // 최대 30개로 제한
    if (history.length > 30) {
      history = history.sublist(0, 30);
    }

    await prefs.setStringList(_key, history);
  }

  // 검색어 불러오기
  static Future<List<String>> getSearchHistory() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getStringList(_key) ?? [];
  }

  // 검색어 전체 삭제
  static Future<void> clearSearchHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }

  // 개별 검색어 삭제 메서드 추가
  static Future<void> removeSearchTerm(String query) async {
    final prefs = await SharedPreferences.getInstance();

    // 기존 검색 히스토리 불러오기
    List<String> history = prefs.getStringList(_key) ?? [];

    // 해당 검색어 삭제
    history.remove(query);

    // 업데이트된 히스토리 저장
    await prefs.setStringList(_key, history);
  }
}
