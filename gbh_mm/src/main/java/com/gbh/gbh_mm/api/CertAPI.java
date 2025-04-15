package com.gbh.gbh_mm.api;

import com.gbh.gbh_mm.user.model.request.CIRequestDto;
import com.gbh.gbh_mm.user.model.request.CertExistRequestDto;
import com.gbh.gbh_mm.user.model.request.CertIssueRequestDto;
import com.gbh.gbh_mm.user.model.request.DigitalSignatureIssueRequestDto;
import com.gbh.gbh_mm.user.model.response.CIResponseDto;
import com.gbh.gbh_mm.user.model.response.CertExistResponseDto;
import com.gbh.gbh_mm.user.model.response.CertResponseDto;
import com.gbh.gbh_mm.user.model.response.DigitalSignatureIssueResponseDto;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Component
public class CertAPI {

    private final WebClient webClient;

    @Value("${ssafy.api-key}")
    private String API_KEY;

    public CertAPI(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder
                .baseUrl("http://yun-server.duckdns.org:9001")
//                .baseUrl("http://localhost:9001")
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .build();
    }

    public CIResponseDto createConnectionInformation(CIRequestDto ciRequestDto){
        return webClient.post()
                .uri("/api/cert/ci")
                .bodyValue(ciRequestDto)
                .retrieve()
                .bodyToMono(CIResponseDto.class)
                .block();
    }


    public CertResponseDto createCertificate(CertIssueRequestDto certIssueRequestDto) {
        return webClient.post()
                .uri("/api/cert/issue")
                .bodyValue(certIssueRequestDto)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, res ->
                        res.bodyToMono(String.class).flatMap(body -> {
                            return Mono.error(new RuntimeException(body)); // ✨ cert가 준 메시지 그대로!
                        }))
                .onStatus(HttpStatusCode::is5xxServerError, res ->
                        res.bodyToMono(String.class).flatMap(body -> {
                            return Mono.error(new RuntimeException("cert 서버 장애: " + body));
                        }))
                .bodyToMono(CertResponseDto.class)
                .block();
    }

    public CertExistResponseDto checkCertificateExistence(CertExistRequestDto certExistRequestDto) {
        return webClient.post()
                .uri("/api/cert/exist")
                .bodyValue(certExistRequestDto)  // JSON 본문으로 전송
                .retrieve()
                .bodyToMono(CertExistResponseDto.class)
                .block();
    }

    public DigitalSignatureIssueResponseDto createDigitalSignature(DigitalSignatureIssueRequestDto digitalSignatureIssueRequestDto) {
        return webClient.post()
                .uri("/api/cert/digital-signature")
                .bodyValue(digitalSignatureIssueRequestDto)  // JSON 본문으로 전송
                .retrieve()
                .bodyToMono(DigitalSignatureIssueResponseDto.class)
                .block();
    }
}
