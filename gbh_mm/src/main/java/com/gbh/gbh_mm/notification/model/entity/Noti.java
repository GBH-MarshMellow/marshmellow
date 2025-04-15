package com.gbh.gbh_mm.notification.model.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Duration;
import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Noti {
    private String title;
    private String body;
    private LocalDateTime receivedAt;

    public String getTimeAgo() {
        Duration duration = Duration.between(receivedAt, LocalDateTime.now());
        long minutes = duration.toMinutes();
        if (minutes < 1) return "방금 전";
        if (minutes < 60) return minutes + "분 전";
        long hours = duration.toHours();
        if (hours < 24) return hours + "시간 전";
        long days = duration.toDays();
        return days + "일 전";
    }

}
