package com.gbh.gbh_mm.api;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCheckAccountAuth;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCreateAccountAuth;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class AuthAPI {
    private final WebClient webClient;

    @Value("${ssafy.api-key}")
    private String API_KEY;

    public AuthAPI(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder
                .baseUrl("https://finopenapi.ssafy.io/ssafy/api/v1/edu/accountAuth")
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    public Map<String, Object> getDefaltHeader(String apiName) {
        LocalDateTime now = LocalDateTime.now();
        String date = now.format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        String time = now.format(DateTimeFormatter.ofPattern("HHmmss"));

        System.out.println(date);
        System.out.println(time);

        Map<String, Object> header = new HashMap<>();

        String institutionTransactionUniqueNo =
                date + time
                        + String.format("%06d", (int) (Math.random() * 1000000));

        header.put("apiName", apiName);
        header.put("transmissionDate", date);
        header.put("transmissionTime", time);
        header.put("institutionCode", "00100");
        header.put("fintechAppNo", "001");
        header.put("apiServiceCode", apiName);
        header.put("institutionTransactionUniqueNo", institutionTransactionUniqueNo);
        header.put("apiKey", API_KEY);

        return header;
    }

    public Map<String, Object> createAccountAuth(RequestCreateAccountAuth request)
            throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header = getDefaltHeader("openAccountAuth");
        header.put("userKey", request.getUserKey());

        requestBody.put("Header", header);
        requestBody.put("accountNo", request.getAccountNo());
        requestBody.put("authText", "MarshMellow");


        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
                .uri("/openAccountAuth")
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

    public Map<String, Object> checkAccountAuth(RequestCheckAccountAuth request)
            throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header = getDefaltHeader("checkAuthCode");
        header.put("userKey", request.getUserKey());

        requestBody.put("Header", header);
        requestBody.put("accountNo", request.getAccountNo());
        requestBody.put("authCode", request.getAuthCode());
        requestBody.put("authText", "MarshMellow");


        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
                .uri("/checkAuthCode")
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
