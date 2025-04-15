package com.gbh.marshmellow

import android.content.Intent
import android.os.Bundle
import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugins.GeneratedPluginRegistrant

class MainActivity: FlutterActivity() {
    private val CHANNEL = "app.channel.shared.data"
    private val WIDGET_CHANNEL = "com.gbh.marshmellow/widget"
    private var sharedText: String? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        GeneratedPluginRegistrant.registerWith(flutterEngine)
        
        // 공유 데이터 채널
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "getSharedText") {
                result.success(sharedText)
                sharedText = null
            } else {
                result.notImplemented()
            }
        }
        
        // 위젯 데이터 채널
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, WIDGET_CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "saveWidgetData" -> {
                    try {
                        val args = call.arguments as Map<*, *>
                        val amount = args["amount"] as Int
                        val title = args["title"] as String
                        
                        // 직접 SharedPreferences에 저장
                        val prefs = context.getSharedPreferences("widgetData", Context.MODE_PRIVATE)
                        prefs.edit()
                            .putInt("amount", amount)
                            .putString("title", title)
                            .apply()
                        
                        Log.d("MainActivity", "위젯 데이터 직접 저장: $amount")
                        
                        // 위젯 업데이트 트리거
                        val intent = Intent(context, BudgetWidgetProvider::class.java)
                        intent.action = android.appwidget.AppWidgetManager.ACTION_APPWIDGET_UPDATE
                        val ids = android.appwidget.AppWidgetManager.getInstance(context)
                            .getAppWidgetIds(android.content.ComponentName(context, BudgetWidgetProvider::class.java))
                        intent.putExtra(android.appwidget.AppWidgetManager.EXTRA_APPWIDGET_IDS, ids)
                        context.sendBroadcast(intent)
                        
                        result.success(true)
                    } catch (e: Exception) {
                        Log.e("MainActivity", "위젯 데이터 저장 오류: ${e.message}")
                        result.error("WIDGET_ERROR", "위젯 데이터 저장 실패", e.toString())
                    }
                }
                else -> result.notImplemented()
            }
        }
        
        // 인텐트를 처리합니다
        handleIntent(intent)
    }
    
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }
    
    private fun handleIntent(intent: Intent) {
        val action = intent.action
        val type = intent.type
        
        if (Intent.ACTION_SEND == action && type != null) {
            if ("text/plain" == type) {
                handleSendText(intent)
            }
        }
    }
    
    private fun handleSendText(intent: Intent) {
        intent.getStringExtra(Intent.EXTRA_TEXT)?.let { text ->
            sharedText = text
            
            // 앱이 이미 실행 중이면 MethodChannel을 통해 메시지를 전송
            flutterEngine?.let { engine ->
                MethodChannel(engine.dartExecutor.binaryMessenger, CHANNEL)
                    .invokeMethod("sharedText", text)
            }
        }
    }
}
