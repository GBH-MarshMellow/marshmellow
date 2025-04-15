package com.gbh.marshmellow

import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.Context
import android.content.SharedPreferences
import android.net.Uri  // Uri import 추가
import android.widget.RemoteViews
import android.app.PendingIntent  // PendingIntent import 추가
import android.content.Intent  // Intent import 추가
import android.util.Log
import es.antonborri.home_widget.HomeWidgetLaunchIntent  // 이 클래스 import

class BudgetWidgetProvider : AppWidgetProvider() {

    override fun onUpdate(context: Context, appWidgetManager: AppWidgetManager, appWidgetIds: IntArray) {
        // 모든 위젯 인스턴스 업데이트
        appWidgetIds.forEach { widgetId ->
            updateWidget(context, appWidgetManager, widgetId)
        }
    }
    
    private fun updateWidget(context: Context, appWidgetManager: AppWidgetManager, appWidgetId: Int) {
        try {
            // 위젯 레이아웃 로드
            val views = RemoteViews(context.packageName, R.layout.budget_widget_layout)
            
            // 여러 SharedPreferences 소스에서 데이터 로드 시도
            var amount = 0
            var title = "오늘의 예산"
            
            try {
                // 1. FlutterSharedPreferences에서 먼저 로드 시도
                val flutterPrefs = context.getSharedPreferences("FlutterSharedPreferences", Context.MODE_PRIVATE)
                amount = flutterPrefs.getInt("flutter.amount", 0)
                Log.d("BudgetWidgetProvider", "FlutterSharedPreferences에서 로드: $amount")
                
                // 2. 값이 없으면 widgetData에서 로드 시도
                if (amount == 0) {
                    val widgetPrefs = context.getSharedPreferences("widgetData", Context.MODE_PRIVATE)
                    amount = widgetPrefs.getInt("amount", 0)
                    Log.d("BudgetWidgetProvider", "widgetData에서 로드: $amount")
                }
                
                // 3. 그래도 값이 없으면 일반 SharedPreferences에서 로드 시도
                if (amount == 0) {
                    val normalPrefs = context.getSharedPreferences("prefs", Context.MODE_PRIVATE)
                    amount = normalPrefs.getInt("last_daily_budget", 0)
                    Log.d("BudgetWidgetProvider", "normalPrefs에서 로드: $amount")
                }
            } catch (e: Exception) {
                Log.e("BudgetWidgetProvider", "데이터 로드 오류: ${e.message}")
                // 기본값 설정
                amount = 75305 // 디버그용 기본값
            }
            
            // 데이터 로깅
            Log.d("BudgetWidgetProvider", "최종 로드된 금액: $amount")
            
            // 숫자 천 단위 구분자로 포맷팅
            val formattedAmount = java.text.NumberFormat.getNumberInstance(java.util.Locale.KOREA).format(amount) + "원"
            
            // 위젯 UI 업데이트
            views.setTextViewText(R.id.title_text, "오늘의 예산")
            views.setTextViewText(R.id.amount_text, formattedAmount)
            
            // 위젯 텍스트 색상 설정
            views.setTextColor(R.id.title_text, android.graphics.Color.parseColor("#9E9E9E"))
            views.setTextColor(R.id.amount_text, android.graphics.Color.parseColor("#000000"))
            
            // 로고 이미지 색상 설정 (흰색으로)
            views.setImageViewResource(R.id.logo_image, R.drawable.logo)
            views.setInt(R.id.logo_image, "setColorFilter", android.graphics.Color.WHITE)
            
            // 위젯 클릭 시 앱 실행하는 PendingIntent 설정
            val intent = Intent(context, MainActivity::class.java).apply {
                action = "budgetwidget.open"
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            val pendingIntent = PendingIntent.getActivity(
                context, 
                appWidgetId, 
                intent, 
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
            views.setOnClickPendingIntent(R.id.widget_container, pendingIntent)
            
            // 위젯 업데이트 적용
            appWidgetManager.updateAppWidget(appWidgetId, views)
            Log.d("BudgetWidgetProvider", "위젯 업데이트 완료: $formattedAmount")
        } catch (e: Exception) {
            Log.e("BudgetWidgetProvider", "위젯 업데이트 중 오류 발생: ${e.message}")
        }
    }
    
    // 위젯이 처음 추가될 때 호출
    override fun onEnabled(context: Context) {
        super.onEnabled(context)
        Log.d("BudgetWidgetProvider", "위젯 활성화됨")
    }
    
    // 위젯의 마지막 인스턴스가 제거될 때 호출
    override fun onDisabled(context: Context) {
        super.onDisabled(context)
        Log.d("BudgetWidgetProvider", "위젯 비활성화됨")
    }
    
    // 위젯 데이터 수신
    override fun onReceive(context: Context, intent: Intent) {
        super.onReceive(context, intent)
        
        // HomeWidget에서 보낸 업데이트 인텐트 처리
        if (intent.action == AppWidgetManager.ACTION_APPWIDGET_UPDATE) {
            val appWidgetManager = AppWidgetManager.getInstance(context)
            val appWidgetIds = intent.getIntArrayExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS) ?: return
            
            // 위젯 업데이트
            onUpdate(context, appWidgetManager, appWidgetIds)
            Log.d("BudgetWidgetProvider", "위젯 업데이트 인텐트 수신")
        }
    }
}