package com.gbh.gbh_mm.user.repo;

import org.springframework.stereotype.Repository;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Repository
public class EmitterRepositoryImpl implements EmitterRepository {

    private final Map<String, SseEmitter> emitters = new ConcurrentHashMap<>();
    private final Map<String, Object> eventCache = new ConcurrentHashMap<>();

    @Override
    public SseEmitter save(String emitterId, SseEmitter sseEmitter) { // emitter를 저장
        emitters.put(emitterId, sseEmitter);
        return sseEmitter;
    }

    @Override
    public void saveEventCache(String eventCacheId, Object event) { // 이벤트를 저장
        eventCache.put(eventCacheId, event);
    }

    @Override
    public Map<String, SseEmitter> findAllEmitterStartWithByPhoneNumber(String phoneNumber) { // 해당 번호와 관련된 모든 이벤트를 찾음
        return emitters.entrySet().stream()
                .filter(entry -> entry.getKey().startsWith(phoneNumber))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    @Override
    public Map<String, Object> findAllEventCacheStartWithByPhoneNumber(String phoneNumber) {
        return eventCache.entrySet().stream()
                .filter(entry -> entry.getKey().startsWith(phoneNumber))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    @Override
    public void deleteById(String id) { // emitter를 지움
        emitters.remove(id);
    }

    @Override
    public void deleteAllEventCacheStartWithPhoneNumber(String phoneNumber) { // 해당 번호와 관련된 모든 이벤트를 지움
        eventCache.forEach(
                (key, emitter) -> {
                    if (key.startsWith(phoneNumber)) {
                        eventCache.remove(key);
                    }
                }
        );
    }
}
