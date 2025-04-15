import 'package:flutter/material.dart';
import 'package:marshmellow/core/theme/app_colors.dart';
import 'package:marshmellow/core/theme/app_text_styles.dart';
import 'package:marshmellow/presentation/widgets/custom_appbar/custom_appbar.dart';
import 'package:marshmellow/presentation/pages/etc/notifications/notifications_item.dart';

/*
  알림 페이지 UI
*/
class NotificationsPage extends StatelessWidget{
  final List<NotificationItem>? notifications;

  const NotificationsPage({
    Key? key, 
    this.notifications
    }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // [edit]더미데이터이므로 실제로는 서버에서 받아온 데이터를 사용해야 합니다
    final notificationList = notifications ?? _getTestNotifications();

    return Scaffold(
      appBar: CustomAppbar(
        title: '알림',
        hideNotificaiotnIcon: true,
      ),
      // 스와이프 제스처 감지 뒤로가기
      body: GestureDetector(
        onHorizontalDragEnd: (details) {
          // 좌에서 우로 스와이프 감지
          if (details.primaryVelocity != null && details.primaryVelocity! > 0) {
            Navigator.pop(context);
          }
        },
        child: notificationList.isEmpty
            ? const Center(
                child: Padding(
                  padding: EdgeInsets.all(24.0),
                  child: Text(
                    '확인해야 할 알림이 없습니다',
                    style: TextStyle(
                      color: Colors.grey,
                      fontSize: 16,
                    ),
                  ),
                ),
              )
            : ListView.separated(
                padding: const EdgeInsets.all(16),
                itemCount: notificationList.length,
                separatorBuilder: (context, index) => const Divider(),
                itemBuilder: (context, index) {
                  final notification = notificationList[index];
                  return ListTile(
                    title: Text(
                      notification.title,
                      style: AppTextStyles.bodyMedium.copyWith(
                        color: notification.isRead
                            ? AppColors.textLight
                            : AppColors.textPrimary,
                        ),
                    ),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          notification.message,
                          style: AppTextStyles.bodySmall.copyWith(
                            color: notification.isRead
                                ? AppColors.textLight
                                : AppColors.textPrimary,
                          ),),
                        const SizedBox(height: 4),
                        Text(
                          _formatTime(notification.time),
                          style: const TextStyle(
                            fontSize: 12,
                            color: Colors.grey,
                          ),
                        ),
                      ],
                    ),
                    isThreeLine: true,
                    onTap: () {
                      // 알림 클릭 처리
                    },
                  );
                },
              ),
      ),
    );
  }

  // 알림 더미데이터 입니다
  List<NotificationItem> _getTestNotifications() {
    return [
       NotificationItem(
        title: '오늘의 예산✨',
        message: '오늘은 14,095원만 써보세요!',
        time: DateTime.now().subtract(const Duration(minutes: 5)),
      ),
      NotificationItem(
        title: '경 월급날 축',
        message: '오늘은 기다리던 월급날입니다🎉',
        time: DateTime.now().subtract(const Duration(hours: 2)),
        isRead: true,
      ),
      NotificationItem(
        title: '예산 초과 경고',
        message: '조심하세요! 이번달 식비 예산의 70%를 사용했어요.',
        time: DateTime.now().subtract(const Duration(days: 1)),
      ),
      NotificationItem(
        title: '오늘의 버티기',
        message: '3시간째 일하는 중! 1.6 뿌링클을 벌었어요🐥',
        time: DateTime.now().subtract(const Duration(days: 3)),
        isRead: true,
      ),
    ];
  }

  // 시간 포맷팅
  String _formatTime(DateTime time) {
    final now = DateTime.now();
    final diff = now.difference(time);
    if (diff.inSeconds < 60) {
      return '방금 전';
    } else if (diff.inMinutes < 60) {
      return '${diff.inMinutes}분 전';
    } else if (diff.inHours < 24) {
      return '${diff.inHours}시간 전';
    } else if (diff.inDays < 7) {
      return '${diff.inDays}일 전';
    } else {
      return '${time.year}.${time.month}.${time.day}';
    }
  }

}