package com.gbh.gbh_mm.api;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestCreateDepositProduct;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Component
public class BankAPI {
    private final WebClient webClient;

    @Value("${ssafy.api-key}")
    private String API_KEY;

    public BankAPI(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder
            .baseUrl("https://finopenapi.ssafy.io/ssafy/api/v1/edu/bank")
            .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
            .build();
    }

    /* 1. 은행 목록 조회 */
    public Map<String, Object> findBankList() throws JsonProcessingException {
        // API 요청을 위한 JSON Body
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header = getDefaltHeader("inquireBankCodes");

        requestBody.put("Header", header);

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        // API 요청
        String response = webClient.post()
            .uri("/inquireBankCodes")
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue(requestBody)
            .retrieve()
            .onStatus(status -> status.isError(),
                clientResponse -> clientResponse.bodyToMono(String.class)
                    .flatMap(errorBody -> Mono.error(new RuntimeException("🔥 API 오류 응답: " + errorBody)))
            )
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


}
