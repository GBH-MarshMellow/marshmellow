import 'package:home_widget/home_widget.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:io';
import 'package:flutter/services.dart';

class WidgetService {
  // 기본 위젯 업데이트 메서드
  static Future<void> updateBudgetWidget(int amount) async {
    try {
      // 디버그 로깅 추가
      print('위젯 업데이트 시도: $amount원');

      // 1. SharedPreferences에 값 저장 (백업용)
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt('last_daily_budget', amount);

      // 2. Android 전용 - 일반 SharedPreferences에도 저장 (백업)
      if (Platform.isAndroid) {
        await _saveToAndroidDirectPrefs(amount);
      }

      // 3. HomeWidget 플러그인을 통한 저장 (주요 방식)
      await HomeWidget.saveWidgetData('amount', amount);
      await HomeWidget.saveWidgetData('title', '오늘의 예산');

      // 위젯 업데이트 요청 - 명시적으로 이름 지정
      await HomeWidget.updateWidget(
        androidName: 'BudgetWidgetProvider',
        iOSName: 'BudgetWidgetProvider',
      );

      // 업데이트 확인
      await Future.delayed(Duration(milliseconds: 500));
      final savedAmount = await HomeWidget.getWidgetData<int>('amount');
      print('위젯 업데이트 성공: $amount원 (저장된 값: $savedAmount)');
    } catch (e) {
      print('위젯 업데이트 오류: $e');
      // 오류 발생 시 백업 방식 시도
      await _fallbackWidgetUpdate(amount);
    }
  }

  // Android에서 직접 SharedPreferences에 저장하는 메서드
  static Future<void> _saveToAndroidDirectPrefs(int amount) async {
    try {
      const platform = MethodChannel('com.gbh.marshmellow/widget');
      await platform.invokeMethod('saveWidgetData', {
        'amount': amount,
        'title': '오늘의 예산',
      });
      print('Android 직접 SharedPreferences 저장 성공');
    } catch (e) {
      print('Android 직접 SharedPreferences 저장 오류: $e');
      // 백업 저장 방식으로 SharedPreferences 사용
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt('widget_amount', amount);
    }
  }

  // 백업 위젯 업데이트 메서드 (주 방식 실패 시)
  static Future<void> _fallbackWidgetUpdate(int amount) async {
    try {
      // 1. 다양한 키로 시도
      await HomeWidget.saveWidgetData<int>('widgetAmount', amount);
      await HomeWidget.saveWidgetData<String>('widgetTitle', '오늘의 예산');

      // 2. 문자열로 변환해서 시도
      await HomeWidget.saveWidgetData<String>('amountStr', amount.toString());

      // 3. 위젯 강제 업데이트
      await HomeWidget.updateWidget(
        androidName: 'BudgetWidgetProvider',
        iOSName: 'BudgetWidgetProvider',
      );

      print('백업 위젯 업데이트 시도 완료');
    } catch (e) {
      print('백업 위젯 업데이트 오류: $e');
    }
  }

  // 기본값으로 위젯 업데이트 (앱이 백그라운드일 때 사용)
  static Future<void> updateBudgetWidgetDefaults() async {
    try {
      // SharedPreferences에서 마지막으로 저장된 예산 값 가져오기 (없으면 0)
      final prefs = await SharedPreferences.getInstance();
      final lastAmount = prefs.getInt('last_daily_budget') ?? 0;

      print('기본 위젯 업데이트: 마지막 저장 값 = $lastAmount');

      // 위젯 업데이트
      await updateBudgetWidget(lastAmount);
    } catch (e) {
      print('기본 위젯 업데이트 오류: $e');
      // 오류 발생 시 0원으로 업데이트
      await updateBudgetWidget(0);
    }
  }

  // 위젯 클릭 이벤트 수신 설정
  static void setupWidgetClicks(void Function(Uri?)? onWidgetClicked) {
    HomeWidget.widgetClicked.listen(onWidgetClicked);
  }
}
