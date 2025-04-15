package com.gbh.gbh_mm.api;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateCardProduct;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateMerchant;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateTransaction;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateUserCard;
import com.gbh.gbh_mm.finance.card.vo.request.RequestDeleteTransaction;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindBilling;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindCardTransactionList;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindUserCardList;
import com.gbh.gbh_mm.finance.card.vo.request.RequestUpdateAccount;
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
public class CardAPI {

    private final WebClient webClient;

    @Value("${ssafy.api-key}")
    private String API_KEY;

    public CardAPI(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder
            .baseUrl("https://finopenapi.ssafy.io/ssafy/api/v1/edu/creditCard")
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

    public Map<String, Object> findCategoryList() throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "inquireCategoryList";

        Map<String, Object> header = getDefaltHeader(apiName);

        requestBody.put("Header", header);

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

    public Map<String, Object> createMerchant(RequestCreateMerchant request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "createMerchant";

        Map<String, Object> header = getDefaltHeader(apiName);

        requestBody.put("Header", header);
        requestBody.put("categoryId", request.getCategoryId());
        requestBody.put("merchantName", request.getMerchantName());

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

    public Map<String, Object> findCompanyList()
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "inquireCardIssuerCodesList";

        Map<String, Object> header = getDefaltHeader(apiName);

        requestBody.put("Header", header);

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

    public Map<String, Object> createProduct(RequestCreateCardProduct request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "createCreditCardProduct";

        Map<String, Object> header = getDefaltHeader(apiName);

        requestBody.put("Header", header);
        requestBody.put("cardIssuerCode", request.getCardIssuerCode());
        requestBody.put("cardName", request.getCardName());
        requestBody.put("baselinePerformance", request.getBaselinePerformance());
        requestBody.put("maxBenefitLimit", request.getMaxBenefitLimit());
        requestBody.put("cardDescription", request.getCardDescription());
        requestBody.put("cardBenefits", request.getCardBenefits());

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

    public Map<String, Object> findProductList()
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "inquireCreditCardList";

        Map<String, Object> header = getDefaltHeader(apiName);

        requestBody.put("Header", header);

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

    public Map<String, Object> createUserCard(RequestCreateUserCard request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "createCreditCard";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());

        requestBody.put("Header", header);
        requestBody.put("cardUniqueNo", request.getCardUniqueNo());
        requestBody.put("withdrawalAccountNo", request.getWithdrawalAccountNo());
        requestBody.put("withdrawalDate", request.getWithdrawalDate());

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

    public Map<String, Object> findUserCardList(String userKey) throws JsonProcessingException {
        // 요청 본문 구성
        Map<String, Object> requestBody = new HashMap<>();
        String apiName = "inquireSignUpCreditCardList";
        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", userKey);
        requestBody.put("Header", header);

        System.out.println(requestBody);

        // 버츄얼 스레드용 Executor 생성
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<Map<String, Object>> future = executor.submit(() -> {
                // WebClient를 통한 블로킹 호출
                String response = webClient.post()
                    .uri("/" + apiName)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(requestBody)
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();  // 블로킹 호출
                // JSON 파싱
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


    public Map<String, Object> findMerchantList() throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "inquireMerchantList";

        Map<String, Object> header = getDefaltHeader(apiName);

        requestBody.put("Header", header);

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

    public Map<String, Object> createTransaction(RequestCreateTransaction request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "createCreditCardTransaction";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());

        requestBody.put("Header", header);
        requestBody.put("cardNo", request.getCardNo());
        requestBody.put("cvc", request.getCvc());
        requestBody.put("merchantId", request.getMerchantId());
        requestBody.put("paymentBalance", request.getPaymentBalance());

        System.out.println(requestBody);

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

    public Map<String, Object> findTransactionList(RequestFindCardTransactionList request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "inquireCreditCardTransactionList";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());

        requestBody.put("Header", header);
        requestBody.put("cardNo", request.getCardNo());
        requestBody.put("cvc", request.getCvc());
        requestBody.put("startDate", request.getStartDate());
        requestBody.put("endDate", request.getEndDate());

        System.out.println(requestBody);

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

    public Map<String, Object> deleteTransaction(RequestDeleteTransaction request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "deleteTransaction";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());

        requestBody.put("Header", header);
        requestBody.put("cardNo", request.getCardNo());
        requestBody.put("cvc", request.getCvc());
        requestBody.put("transactionUniqueNo", request.getTransactionUniqueNo());

        System.out.println(requestBody);

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

    public Map<String, Object> findBilling(RequestFindBilling request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "inquireBillingStatements";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());

        requestBody.put("Header", header);
        requestBody.put("cardNo", request.getCardNo());
        requestBody.put("cvc", request.getCvc());
        requestBody.put("startMonth", request.getStartMonth());
        requestBody.put("endMonth", request.getEndMonth());

        System.out.println(requestBody);

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

    public Map<String, Object> updateAccount(RequestUpdateAccount request)
        throws JsonProcessingException {
        Map<String, Object> requestBody = new HashMap<>();

        String apiName = "updateWithdrawalAccount";

        Map<String, Object> header = getDefaltHeader(apiName);
        header.put("userKey", request.getUserKey());

        requestBody.put("Header", header);
        requestBody.put("cardNo", request.getCardNo());
        requestBody.put("cvc", request.getCvc());
        requestBody.put("withdrawalAccountNo", request.getWithdrawalAccountNo());
        requestBody.put("withdrawalDate", request.getWithdrawalDate());

        System.out.println(requestBody);

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
