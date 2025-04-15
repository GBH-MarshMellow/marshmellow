package com.gbh.gbh_mm.user.repo;

import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Map;

public interface EmitterRepository {
    SseEmitter save(String emitterId, SseEmitter sseEmitter);

    void saveEventCache(String eventCacheId, Object event);

    Map<String, SseEmitter> findAllEmitterStartWithByPhoneNumber(String phoneNumber);

    Map<String, Object> findAllEventCacheStartWithByPhoneNumber(String phoneNumber);

    void deleteById(String id);

    void deleteAllEventCacheStartWithPhoneNumber(String phoneNumber);
}
