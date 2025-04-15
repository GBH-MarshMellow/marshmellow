package com.gbh.gbh_mm.api;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestCreateAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestCreateDepositProduct;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestDeleteAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestEarlyInterest;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccountList;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindInterest;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindPayment;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;

@Component
public class DepositAPI {
    private final WebClient webClient;

    @Value("${ssafy.api-key}")
    private String API_KEY;

    public DepositAPI(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder
            .baseUrl("https://finopenapi.ssafy.io/ssafy/api/v1/edu/deposit")
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


    public Map<String, Object> createDepositProduct(RequestCreateDepositProduct request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header = getDefaltHeader("createDepositProduct");

        requestBody.put("Header", header);
        requestBody.put("bankCode", request.getBankCode());
        requestBody.put("accountName", request.getAccountName());
        requestBody.put("accountDescription", request.getAccountDescription());
        requestBody.put("subscriptionPeriod", request.getSubscriptionPeriod());
        requestBody.put("maxSubscriptionBalance", request.getMaxSubscriptionBalance());
        requestBody.put("minSubscriptionBalance", request.getMinSubscriptionBalance());
        requestBody.put("interestRate", request.getInterestRate());
        requestBody.put("rateDescription", request.getRateDescription());


        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
            .uri("/createDepositProduct")
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

    public Map<String, Object> findProductList() throws JsonProcessingException {

        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header = getDefaltHeader("inquireDepositProducts");

        requestBody.put("Header", header);

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
            .uri("/inquireDepositProducts")
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

    public Map<String, Object> createAccount(RequestCreateAccount request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header = getDefaltHeader("createDepositAccount");
        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);

        requestBody.put("withdrawalAccountNo", request.getWithdrawalAccountNo());
        requestBody.put("accountTypeUniqueNo", request.getAccountTypeUniqueNo());
        requestBody.put("depositBalance", request.getDepositBalance());

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
            .uri("/createDepositAccount")
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

    public Map<String, Object> findAccountList(String userKey) throws JsonProcessingException {
        // 요청 본문 구성
        Map<String, Object> requestBody = new HashMap<>();
        String apiName = "inquireDepositInfoList";
        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", userKey);
        requestBody.put("Header", header);

        // 버츄얼 스레드를 위한 Executor 생성 (Java 21)
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<Map<String, Object>> future = executor.submit(() -> {
                // WebClient를 통한 블로킹 호출 (버츄얼 스레드에서 실행)
                String response = webClient.post()
                    .uri("/" + apiName)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(requestBody)
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();  // 블로킹 호출이지만 버츄얼 스레드에서 처리됨

                // 응답 문자열을 JSON(Map)으로 변환
                ObjectMapper objectMapper = new ObjectMapper();
                Map<String, Object> apiResponseJson = objectMapper.readValue(response, Map.class);

                // 성공 응답 JSON 구성
                Map<String, Object> responseJson = new LinkedHashMap<>();
                responseJson.put("status", "success");
                responseJson.put("apiResponse", apiResponseJson);
                return responseJson;
            });
            return future.get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }


    public Map<String, Object> findAccount(RequestFindAccount request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "inquireDepositInfoDetail";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);

        requestBody.put("accountNo", request.getAccountNo());

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
            .uri("/" + apiName)
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

    public Map<String, Object> findPayment(RequestFindPayment request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "inquireDepositPayment";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);

        requestBody.put("accountNo", request.getAccountNo());

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
            .uri("/" + apiName)
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

    public Map<String, Object> findInterest(RequestFindInterest request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "inquireDepositExpiryInterest";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);

        requestBody.put("accountNo", request.getAccountNo());

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
            .uri("/" + apiName)
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

    public Map<String, Object> findEarlyInterest(RequestEarlyInterest request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "inquireDepositEarlyTerminationInterest";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);

        requestBody.put("accountNo", request.getAccountNo());

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
            .uri("/" + apiName)
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

    public Map<String, Object> deleteAccount(RequestDeleteAccount request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "deleteDepositAccount";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);

        requestBody.put("accountNo", request.getAccountNo());

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
            .uri("/" + apiName)
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
