package com.gbh.gbh_mm.api;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestAccountTransfer;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestCreateDemandDepositAccount;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestCreateDepositProduct;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestDeleteDemandDepositAccount;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestDemandDepositDeposit;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestDemandDepositWithdrawal;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindBalance;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindDemandDepositAccount;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindDemandDepositAccountList;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindHolderName;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindTransactionList;
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
public class DemandDepositAPI {

    private final WebClient webClient;

    @Value("${ssafy.api-key}")
    private String API_KEY;

    public DemandDepositAPI(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder
            .baseUrl("https://finopenapi.ssafy.io/ssafy/api/v1/edu/demandDeposit")
            .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
            .build();
    }

    /* 0. 공통 Header 생성 */
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

    /* 1. 입출금 상품 조회 */
    public Map<String, Object> findDepositList() throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header = getDefaltHeader("inquireDemandDepositList");

        requestBody.put("Header", header);

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
            .uri("/inquireDemandDepositList")
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

    /* 2. 입출금 상품 등록 */
    public Map<String, Object> createDepositProduct(RequestCreateDepositProduct request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header = getDefaltHeader("createDemandDeposit");

        requestBody.put("Header", header);
        requestBody.put("bankCode", request.getBankCode());
        requestBody.put("accountName", request.getAccountName());
        requestBody.put("accountDescription", request.getAccountDescription());

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        String response = webClient.post()
            .uri("/createDemandDeposit")
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

    /* 3. 입출금 계좌 생성 */
    public Map<String, Object> createDemandDepositAccount(
        RequestCreateDemandDepositAccount request) throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header = getDefaltHeader("createDemandDepositAccount");
        header.put("userKey", request.getUserKey());

        requestBody.put("Header", header);
        requestBody.put("accountTypeUniqueNo", request.getAccountTypeUniqueNo());

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        System.out.println(requestBody);

        String response = webClient.post()
            .uri("/createDemandDepositAccount")
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

    /* 4. 입출금 계좌 목록 */
    public Map<String, Object> findDemandDepositAccountList(String userKey) throws JsonProcessingException {
        // 요청 본문 구성
        Map<String, Object> requestBody = new HashMap<>();
        Map<String, Object> header = getDefaltHeader("inquireDemandDepositAccountList");
        header.put("userKey", userKey);
        requestBody.put("Header", header);

        System.out.println(requestBody);

        // 버츄얼 스레드를 위한 Executor 생성 (Java 21)
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            // 블로킹 호출을 버츄얼 스레드에서 실행
            Future<Map<String, Object>> future = executor.submit(() -> {
                String response = webClient.post()
                    .uri("/inquireDemandDepositAccountList")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(requestBody)
                    .retrieve()
                    .bodyToMono(String.class)
                    .block(); // 블로킹 호출
                ObjectMapper objectMapper = new ObjectMapper();
                Map<String, Object> apiResponseJson = objectMapper.readValue(response, Map.class);
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


    public Map<String, Object> findDemandDepositAccount(RequestFindDemandDepositAccount request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header = getDefaltHeader("inquireDemandDepositAccount");
        header.put("userKey", request.getUserKey());
        requestBody.put("accountNo", request.getAccountNo());

        requestBody.put("Header", header);

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        System.out.println(requestBody);

        String response = webClient.post()
            .uri("/inquireDemandDepositAccount")
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

    public Map<String, Object> findHolderName(RequestFindHolderName request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header =
            getDefaltHeader("inquireDemandDepositAccountHolderName");
        header.put("userKey", request.getUserKey());
        requestBody.put("accountNo", request.getAccountNo());

        requestBody.put("Header", header);

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        System.out.println(requestBody);

        String response = webClient.post()
            .uri("/inquireDemandDepositAccountHolderName")
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

    public Map<String, Object> findBalance(RequestFindBalance request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header =
            getDefaltHeader("inquireDemandDepositAccountBalance");
        header.put("userKey", request.getUserKey());
        requestBody.put("accountNo", request.getAccountNo());

        requestBody.put("Header", header);

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        System.out.println(requestBody);

        String response = webClient.post()
            .uri("/inquireDemandDepositAccountBalance")
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

    public Map<String, Object> withdrawal(RequestDemandDepositWithdrawal request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header =
            getDefaltHeader("updateDemandDepositAccountWithdrawal");
        requestBody.put("accountNo", request.getAccountNo());
        requestBody.put("transactionBalance", request.getTransactionBalance());
        requestBody.put("transactionSummary", request.getTransactionSummary());

        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);

        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        System.out.println(requestBody);

        String response = webClient.post()
            .uri("/updateDemandDepositAccountWithdrawal")
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

    public Map<String, Object> deposit(RequestDemandDepositDeposit request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header =
            getDefaltHeader("updateDemandDepositAccountDeposit");
        requestBody.put("accountNo", request.getAccountNo());
        requestBody.put("transactionBalance", request.getTransactionBalance());
        requestBody.put("transactionSummary", request.getTransactionSummary());

        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);


        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        System.out.println(requestBody);

        String response = webClient.post()
            .uri("/updateDemandDepositAccountDeposit")
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

    public Map<String, Object> accountTransfer(RequestAccountTransfer request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header =
            getDefaltHeader("updateDemandDepositAccountTransfer");
        requestBody.put("depositAccountNo", request.getDepositAccountNo());
        requestBody.put("depositTransactionSummary", request.getDepositTransactionSummary());
        requestBody.put("transactionBalance", request.getTransactionBalance());
        requestBody.put("withdrawalAccountNo", request.getWithdrawalAccountNo());
        requestBody.put("withdrawalTransactionSummary", request.getWithdrawalTransactionSummary());

        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);


        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        System.out.println(requestBody);

        String response = webClient.post()
            .uri("/updateDemandDepositAccountTransfer")
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

    public Map<String, Object> findTransactionList(RequestFindTransactionList request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header =
            getDefaltHeader("inquireTransactionHistoryList");
        requestBody.put("accountNo", request.getAccountNo());
        requestBody.put("startDate", request.getStartDate());
        requestBody.put("endDate", request.getEndDate());
        requestBody.put("transactionType", request.getTransactionType());
        requestBody.put("orderByType", request.getOrderByType());

        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);


        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        System.out.println(requestBody);

        String response = webClient.post()
            .uri("/inquireTransactionHistoryList")
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

    public Map<String, Object> deleteDemandDepositAccount(RequestDeleteDemandDepositAccount request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        Map<String, Object> header =
            getDefaltHeader("deleteDemandDepositAccount");
        requestBody.put("accountNo", request.getAccountNo());
        requestBody.put("refundAccountNo", request.getRefundAccountNo());

        header.put("userKey", request.getUserKey());
        requestBody.put("Header", header);


        Map<String, Object> responseJson = new LinkedHashMap<>(); // 반환할 JSON 객체

        System.out.println(requestBody);

        String response = webClient.post()
            .uri("/deleteDemandDepositAccount")
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
