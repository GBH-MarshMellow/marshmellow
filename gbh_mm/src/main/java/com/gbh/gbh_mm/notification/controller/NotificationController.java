package com.gbh.gbh_mm.notification.controller;


import com.gbh.gbh_mm.notification.service.NotificationService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;


@RestController
@RequiredArgsConstructor
@RequestMapping("/mm/notification")
public class NotificationController {

    private final NotificationService notificationService;

    @GetMapping("/{userPk}")
    public List<Object> getNotifications(@PathVariable("userPk") String userPk) {
        return notificationService.getNotifications(userPk);
    }

    @DeleteMapping("/{userPk}")
    public String deleteNotification(@PathVariable("userPk") String userPk) {
        notificationService.deleteNotifications(userPk);
        return "redis 삭제 완료";
    }
}
