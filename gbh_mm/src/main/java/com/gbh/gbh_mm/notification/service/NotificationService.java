package com.gbh.gbh_mm.notification.service;

import com.gbh.gbh_mm.notification.model.entity.Noti;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
public class NotificationService {

    private final RedisTemplate<String, Object> redisTemplate;

    // 알림 저장 (리스트에 push, TTL은 키가 없을 경우에만 설정)
    public void saveNotification(String userId, Noti noti) {
        String key = "notification:" + userId;
        redisTemplate.opsForList().rightPush(key, noti);

        // 만약 TTL이 없다면 설정 (최초 1회만)
        Long ttl = redisTemplate.getExpire(key, TimeUnit.SECONDS);
        if (ttl == null || ttl == -1) {
            redisTemplate.expire(key, Duration.ofHours(48));
        }
    }

    // 알림 리스트 가져오기
    public List<Object> getNotifications(String userId) {
        String key = "notification:" + userId;
        List<Object> notifications = redisTemplate.opsForList().range(key, 0, -1);

        if (notifications != null) {
            Collections.reverse(notifications);
        }

        return notifications;
    }
    // 알림 삭제
    public void deleteNotifications(String userId) {
        redisTemplate.delete("notification:" + userId);
    }

    public void saveToredis(String userPk, String title, String body) {
        // redis 저장
        saveNotification(userPk, new Noti(title,body,LocalDateTime.now()));
        System.out.println(">>> 알림 저장: " + userPk + ", " + title);

    }
}
