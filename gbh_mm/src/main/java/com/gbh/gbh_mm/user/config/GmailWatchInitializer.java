package com.gbh.gbh_mm.user.config;

import com.gbh.gbh_mm.user.service.GmailService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

import java.io.IOException;

@Component
@RequiredArgsConstructor
@Slf4j
public class GmailWatchInitializer implements ApplicationRunner {

    private final GmailService gmailService;

    @Override
    public void run(ApplicationArguments args) {
        try {
            gmailService.restartWatch(); // ✅ 변경된 메서드 사용
            log.info("✅ Gmail Watch 최초 등록 완료");
        } catch (Exception e) {
            log.error("❌ Gmail Watch 초기 설정 실패", e);
        }
    }
}
