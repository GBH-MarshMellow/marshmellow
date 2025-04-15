package com.gbh.gbh_mm.fcm.service;

import com.gbh.gbh_mm.fcm.model.request.RequestFCMSend;
import com.gbh.gbh_mm.notification.model.entity.Noti;
import com.gbh.gbh_mm.notification.service.NotificationService;
import com.google.firebase.messaging.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.concurrent.ExecutionException;


@Service
@RequiredArgsConstructor
public class FCMService {

    private final NotificationService notificationService;

    // 실제 값 넣어서 알림
    public String sendNotification(Long userPk, RequestFCMSend requestFCMSend) throws InterruptedException, ExecutionException {
        Message message = Message.builder()
                .setToken(requestFCMSend.getToken())
                .setNotification(Notification.builder()
                        .setTitle(requestFCMSend.getTitle())
                        .setBody(requestFCMSend.getBody())
                        .build())
                .setAndroidConfig(AndroidConfig.builder()
                        .setPriority(AndroidConfig.Priority.HIGH)
                        .build())
                .build();
        String response = FirebaseMessaging.getInstance().sendAsync(message).get();
        System.out.println("✅ Successfully sent message: " + response);

        // redis 저장
        notificationService.saveToredis(String.valueOf(userPk),
                requestFCMSend.getTitle(),
                requestFCMSend.getBody());

        return response;

    }

    // AlertService -> fcmService
    public String sendNotification(Message message) throws InterruptedException, ExecutionException {
        String response = FirebaseMessaging.getInstance().sendAsync(message).get();
        System.out.println("✅ Successfully sent message: " + response);
        return response;

    }
}