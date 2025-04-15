package com.gbh.gbh_mm.user.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.gbh.gbh_mm.user.model.response.IdentityVerificationResponseDto;
import com.google.api.client.googleapis.auth.oauth2.GoogleCredential;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.api.services.gmail.Gmail;
import com.google.api.services.gmail.model.*;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class GmailService {

    private Gmail gmail;

    private final RedisTemplate<String, Object> redisTemplate;
    private final EmitterService emitterService;
    @Value("${google.client-id}")
    private String clientId;

    @Value("${google.client-secret}")
    private String clientSecret;

    @Value("${google.refresh-token}")
    private String refreshToken;

    @Value("${google.project-id}")
    private String projectId;

    @Value("${google.pubsub-topic}")
    private String pubSubTopic;

    @PostConstruct
    public void init() {
        try {
            GoogleCredential credential = new GoogleCredential.Builder()
                    .setTransport(GoogleNetHttpTransport.newTrustedTransport())
                    .setJsonFactory(JacksonFactory.getDefaultInstance())
                    .setClientSecrets(clientId, clientSecret)
                    .build()
                    .setRefreshToken(refreshToken);

            // access_token 자동 갱신
            credential.refreshToken();

            this.gmail = new Gmail.Builder(
                    GoogleNetHttpTransport.newTrustedTransport(),
                    JacksonFactory.getDefaultInstance(),
                    credential
            )
                    .setApplicationName("My Gmail Notifier")
                    .build();

            System.out.println("✅ Gmail client initialized");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void startWatch() throws IOException {
        String topicName = String.format("projects/%s/topics/%s", projectId, pubSubTopic);

        WatchRequest watchRequest = new WatchRequest()
                .setTopicName(topicName)
                .setLabelIds(Collections.singletonList("INBOX"));

        WatchResponse response = gmail.users().watch("me", watchRequest).execute();

        // ✅ 여기에 추가
        redisTemplate.opsForValue().set("gmail:lastHistoryId", response.getHistoryId().toString());

        log.info("✅ Gmail watch started! History ID: {}", response.getHistoryId());
    }

    public void restartWatch() {
        try {
            // 기존 watch 제거
            gmail.users().stop("me").execute();
            log.info("🛑 기존 Gmail watch 중단됨");

            // 새로 watch 등록
            startWatch();

        } catch (IOException e) {
            log.error("❌ Gmail watch 재등록 실패", e);
        }
    }

    public ResponseEntity<String> handlePubSubMessage(Map<String, Object> pubsubBody) {
        try {
            // Pub/Sub 메시지 디코딩
            Map<String, Object> message = (Map<String, Object>) pubsubBody.get("message");
            String data = (String) message.get("data");
            String decodedJson = new String(Base64.getDecoder().decode(data));
            log.info("📨 받은 Pub/Sub 메시지: {}", decodedJson);

            // JSON 파싱 및 historyId 추출
            ObjectMapper objectMapper = new ObjectMapper();
            Map<String, Object> map = objectMapper.readValue(decodedJson, Map.class);
            String historyIdStr = String.valueOf(map.get("historyId"));

            // 📌 1. Redis에서 마지막 처리한 historyId 가져오기
            String savedHistoryId = (String) redisTemplate.opsForValue().get("gmail:lastHistoryId");
            BigInteger startHistoryId = savedHistoryId != null
                    ? new BigInteger(savedHistoryId).subtract(BigInteger.ONE)
                    : new BigInteger(historyIdStr);

            // 📌 2. Gmail 히스토리 조회
            ListHistoryResponse response = gmail.users().history().list("me")
                    .setStartHistoryId(startHistoryId)
                    .execute();
            log.info("📒 Gmail history 조회 결과: historyId={}, size={}",
                    response.getHistoryId(), response.getHistory() != null ? response.getHistory().size() : 0);

            if (response.getHistory() == null || response.getHistory().isEmpty()) {
                log.warn("📭 Gmail 히스토리에 새로운 메시지 없음. startHistoryId={}", startHistoryId);
                return ResponseEntity.ok("OK - 새 메시지 없음");
            }

            for (History h : response.getHistory()) {
                List<HistoryMessageAdded> added = h.getMessagesAdded();
                if (added != null) {
                    for (HistoryMessageAdded addedMsg : added) {
                        Message msg = gmail.users().messages().get("me", addedMsg.getMessage().getId()).execute();

                        String from = getHeader(msg, "From");
                        String phoneNumber = extractPhoneNumber(from);

                        String subject = getHeader(msg, "Subject");
                        String emailBody = extractPlainText(msg);
                        log.info("💌 새 메일 수신: phone={} subject={} body={}", phoneNumber, subject, emailBody);
                        Object raw = redisTemplate.opsForValue().get(phoneNumber);
                        IdentityVerificationResponseDto redisData = objectMapper.convertValue(raw, IdentityVerificationResponseDto.class);

                        if (Objects.nonNull(redisData) && !redisData.isVerified()) {
                            if (emailBody.contains(redisData.getCode())) {
                                emitterService.verifyEmail(phoneNumber, redisData.getCode(), 0);
                                log.info("✅ 인증 성공 및 SSE 전송 완료: {}", phoneNumber);
                            } else {
                                log.warn("❌ 인증 실패: 코드 불일치. 입력={}, 저장={}", emailBody, redisData.getCode());
                            }
                        } else {
                            log.warn("❌ 인증 정보 없음 or 이미 인증됨: {}", phoneNumber);
                        }
                    }
                }
            }

            // 📌 3. 마지막 historyId 저장 (마지막에!)
            if (response.getHistoryId() != null) {
                BigInteger newHistoryId = response.getHistoryId();
                BigInteger previousId = savedHistoryId != null
                        ? new BigInteger(savedHistoryId)
                        : BigInteger.ZERO;

                if (newHistoryId.compareTo(previousId) > 0) {
                    redisTemplate.opsForValue().set("gmail:lastHistoryId", newHistoryId.toString());
                    log.info("📦 마지막 historyId 갱신 완료: {}", newHistoryId);
                } else {
                    log.info("⚠️ 기존 historyId({})보다 작거나 같아서 갱신하지 않음", previousId);
                }
            }

            return ResponseEntity.ok("OK");
        } catch (Exception e) {
            log.error("❌ Webhook 처리 실패", e);
            return ResponseEntity.status(500).body("Webhook 처리 실패");
        }
    }

    private String getHeader(Message message, String name) {
        return message.getPayload().getHeaders().stream()
                .filter(h -> h.getName().equalsIgnoreCase(name))
                .map(MessagePartHeader::getValue)
                .findFirst()
                .orElse("(no header)");
    }

    private String extractPhoneNumber(String fromHeader) {
        // 이메일 주소 앞의 숫자만 추출 (정규식으로 010으로 시작하는 번호 찾기)
        if (fromHeader == null) return "";
        return fromHeader.replaceAll(".*?(\\d{11}).*", "$1");  // ex: "01012345678" 추출
    }

    private String extractPlainText(Message message) {
        try {
            MessagePart payload = message.getPayload();
            if ("text/plain".equalsIgnoreCase(payload.getMimeType())) {
                return decodeBody(payload.getBody().getData());
            }

            if (payload.getParts() != null) {
                for (MessagePart part : payload.getParts()) {
                    if ("text/plain".equalsIgnoreCase(part.getMimeType())) {
                        return decodeBody(part.getBody().getData());
                    }
                }
            }
        } catch (Exception e) {
            log.error("본문 파싱 실패", e);
        }
        return "(본문 없음)";
    }

    private String decodeBody(String data) {
        return new String(Base64.getUrlDecoder().decode(data), StandardCharsets.UTF_8);
    }
}

