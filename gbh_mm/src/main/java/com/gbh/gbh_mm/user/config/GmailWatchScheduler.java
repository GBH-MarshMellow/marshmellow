package com.gbh.gbh_mm.user.config;

import com.gbh.gbh_mm.user.service.GmailService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
@RequiredArgsConstructor
@Slf4j
public class GmailWatchScheduler {

    private final GmailService gmailService;

    @Scheduled(fixedRate = 6 * 24 * 60 * 60 * 1000) // 6일마다 실행 (밀리초 단위)
    public void renewGmailWatch() {
        gmailService.restartWatch();
        log.info("🔄 Gmail Watch 재설정 완료");
    }
}
