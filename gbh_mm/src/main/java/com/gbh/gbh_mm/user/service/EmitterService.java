package com.gbh.gbh_mm.user.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.gbh.gbh_mm.user.model.response.IdentityVerificationResponseDto;
import com.gbh.gbh_mm.user.repo.EmitterRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;

@Service
@Slf4j
@RequiredArgsConstructor
public class EmitterService {

    private final EmitterRepository emitterRepository;
    private final RedisTemplate<String, Object> redisTemplate;
    private final ObjectMapper objectMapper;
    // SSE 연결 지속 시간 설정
    private static final Long DEFAULT_TIMEOUT = 60L * 1000 * 4;

    public SseEmitter subscribe(String phoneNumber, String lastEventId) {
        String emitterId = makeTimeIncludeId(phoneNumber);
        SseEmitter emitter = emitterRepository.save(emitterId, new SseEmitter(DEFAULT_TIMEOUT));
        // onCompletion() : 모든 데이터가 정상적으로 전송되면 호출됩니다.
        emitter.onCompletion(() -> emitterRepository.deleteById(emitterId));
        // onTimeout() : 유효 시간이 만료되면 호출됩니다. 클라이언트의 활동이 감지 되지 않는 경우를 의미합니다.
        emitter.onTimeout(() -> emitterRepository.deleteById(emitterId));

        String eventId = makeTimeIncludeId(phoneNumber);
        sendNotification(emitter, eventId, emitterId, "EventStream Created. [phoneNumber=" + phoneNumber + "]");

        // (1-6) 클라이언트가 미수신한 Event 목록이 존재할 경우 전송하여 Event 유실을 예방
        if (hasLostData(lastEventId)) {
            sendLostData(lastEventId, phoneNumber, emitterId, emitter);
        }
        return emitter;
    }

    private void sendNotification(SseEmitter emitter, String eventId, String emitterId, Object data) {
        try {
            emitter.send(SseEmitter.event()
                    .id(eventId)
                    .name("sse")
                    .data(data)
            );
        } catch (IOException exception) {
            emitterRepository.deleteById(emitterId);
        }
    }

    private boolean hasLostData(String lastEventId) { // (5)
        return !lastEventId.isEmpty();
    }

    private void sendLostData(String lastEventId, String phoneNumber, String emitterId, SseEmitter emitter) {
        Map<String, Object> eventCaches = emitterRepository.findAllEventCacheStartWithByPhoneNumber(String.valueOf(phoneNumber));
        eventCaches.entrySet().stream()
                .filter(entry -> lastEventId.compareTo(entry.getKey()) < 0)
                .forEach(entry -> sendNotification(emitter, entry.getKey(), emitterId, entry.getValue()));
    }

    private String makeTimeIncludeId(String phoneNumber) {
        return phoneNumber + "_" + System.currentTimeMillis();
    }

    public void send(String phoneNumber, String authMessage) {
        String eventId = phoneNumber + " _ " + System.currentTimeMillis();
        Map<String, SseEmitter> emitters = emitterRepository.findAllEmitterStartWithByPhoneNumber(phoneNumber);
        emitters.forEach((key, emitter) -> {
            emitterRepository.saveEventCache(key, authMessage);
            sendNotification(emitter, eventId, key, authMessage);
            if ("인증이 완료되었습니다.".equals(authMessage)) {
                emitter.complete();
                emitterRepository.deleteAllEventCacheStartWithPhoneNumber(phoneNumber);
            }
        });
    }

    public boolean verifyEmail(String phoneNumber, String code, int currentTime) {
        Object raw = redisTemplate.opsForValue().get(phoneNumber);
        if (raw == null) {
            send(phoneNumber, "인증 정보가 존재하지 않습니다.");
            return false;
        }

        IdentityVerificationResponseDto identityVerificationResponseDto =
                objectMapper.convertValue(raw, IdentityVerificationResponseDto.class);

        int expiresIn = identityVerificationResponseDto.getExpiresIn();
        if (identityVerificationResponseDto.getCode().equals(code)) {
            if (currentTime > expiresIn) {
                send(phoneNumber, "인증 코드가 만료되었습니다. 새로운 코드를 요청해주세요.");
                return false;
            }

            redisTemplate.opsForValue().set(phoneNumber + ":verified", true);
            send(phoneNumber, "인증이 완료되었습니다.");
            return true;
        }

        send(phoneNumber, "인증이 실패되었습니다.");
        return false;
    }
}