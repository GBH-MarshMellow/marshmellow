package com.gbh.gbh_mm.presentation.controller;

import com.gbh.gbh_mm.presentation.request.RequestHouseholdForPre;
import com.gbh.gbh_mm.presentation.request.RequestSendAlert;
import com.gbh.gbh_mm.presentation.request.RequestUpdateFcmToken;
import com.gbh.gbh_mm.presentation.service.PresentationService;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import lombok.AllArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/presentation")
@AllArgsConstructor
public class PresentationController {
    private final PresentationService presentationService;

    /* 월급 알림 */
    @GetMapping("/salary-alert")
    public String sendSalaryAlert(
            @RequestBody RequestSendAlert request
    ) {
        return presentationService.sendAlert(request);
    }

    /* 가계부 데이터 등록 */

    @PostMapping("/household")
    public String craeteHousehold(
            @RequestBody RequestHouseholdForPre request
    ) {
        return presentationService.createHousehold(request);
    }

    @PatchMapping("/fcm-token")
    public String sendFcmToken(
        @RequestBody RequestUpdateFcmToken request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        return presentationService.updateFcmToken(request, customUserDetails);
    }
}
