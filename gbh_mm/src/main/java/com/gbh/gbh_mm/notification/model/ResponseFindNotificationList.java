package com.gbh.gbh_mm.notification.model;

import com.gbh.gbh_mm.notification.model.entity.Noti;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@AllArgsConstructor
public class ResponseFindNotificationList {
    private String message;
    private List<NotiList> notificationData;

    @Data
    @Builder
    public static class NotiList {
        private String title;
        private String body;
        private String timeAgo;


    }
}
