package com.gbh.gbh_cert.api;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.gbh.gbh_cert.model.dto.request.RequestCreateUserKey;
import com.gbh.gbh_cert.model.dto.request.RequestReissueUserKey;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class UserAPI {
    private final WebClient webClient;

    @Value("${ssafy.api-key}")
    private String API_KEY;

    public UserAPI(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder
                .baseUrl("https://finopenapi.ssafy.io/ssafy/api/v1/member")
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    public Map<String, Object> createUserKey(RequestCreateUserKey request)
            throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("apiKey", API_KEY);
        requestBody.put("userId", request.getUserId());
        System.out.println("requestBody.get(\"apiKey\") = " + requestBody.get("apiKey"));
        System.out.println("requestBody.get(\"userId\") = " + requestBody.get("userId"));
        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(requestBody)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, res ->
                        res.bodyToMono(String.class).flatMap(body -> {
                            System.out.println("❗ 4xx Error Response: " + body);
                            if (body.contains("이미 존재") || body.contains("중복")) {
                                return Mono.error(new RuntimeException("이미 존재하는 userId입니다."));
                            }
                            return Mono.error(new RuntimeException("기타 클라이언트 에러: " + body));
                        }))
                .onStatus(HttpStatusCode::is5xxServerError, res ->
                        res.bodyToMono(String.class).flatMap(body -> {
                            System.out.println("❗ 5xx Error Response: " + body);
                            return Mono.error(new RuntimeException("서버 에러: " + body));
                        }))
                .bodyToMono(String.class)
                .block(); // ⚠️ 동기 처리
        // ✅ String -> JSON 변환
        ObjectMapper objectMapper = new ObjectMapper();
        Map<String, Object> apiResponseJson = objectMapper.readValue(response, Map.class);
        // 성공 응답 JSON 생성
        responseJson.put("status", "success");
        responseJson.put("apiResponse", apiResponseJson); // ✅ JSON 형태로 변환하여 저장
        return responseJson;
    }

    public Map<String, Object> searchUser(RequestReissueUserKey request)
            throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("apiKey", API_KEY);
        requestBody.put("userId", request.getUserId());

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
                .uri("/search")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(requestBody)
                .retrieve()
                .bodyToMono(String.class)
                .block(); // ⚠️ 동기 처리
        // ✅ String -> JSON 변환
        ObjectMapper objectMapper = new ObjectMapper();
        Map<String, Object> apiResponseJson = objectMapper.readValue(response, Map.class);
        // 성공 응답 JSON 생성
        responseJson.put("status", "success");
        responseJson.put("apiResponse", apiResponseJson); // ✅ JSON 형태로 변환하여 저장
        return responseJson;
    }
}
