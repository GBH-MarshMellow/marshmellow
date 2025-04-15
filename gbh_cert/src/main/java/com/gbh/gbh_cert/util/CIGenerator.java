package com.gbh.gbh_cert.util;

import com.gbh.gbh_cert.model.dto.request.CIRequestDto;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

@Component
public class CIGenerator {

    @Value("${cert.sa-key}")
    private String saKey;

    @Value("${cert.sk-key}")
    private String skKey;

    private byte[] SA;
    private byte[] SK;

    @PostConstruct
    private void init() {
        this.SA = saKey.getBytes(StandardCharsets.UTF_8);
        this.SK = skKey.getBytes(StandardCharsets.UTF_8);
    }

    public String generateCi(CIRequestDto request) {
        try {
            // 1. 입력 문자열 생성
            String input = request.getUserCode() + request.getUserName() + request.getPhoneNumber();
            byte[] inputBytes = input.getBytes(StandardCharsets.UTF_8);

            // 2. 길이 64바이트로 패딩 (0x00으로 채움)
            byte[] padded = new byte[64]; // 64Byte = 512bit
            System.arraycopy(inputBytes, 0, padded, 0, Math.min(inputBytes.length, 64));
            for (int i = inputBytes.length; i < 64; i++) {
                padded[i] = 0x00;
            }

            // 3. padded ⊕ SA
            byte[] xored = new byte[64];
            for (int i = 0; i < 64; i++) {
                xored[i] = (byte) (padded[i] ^ SA[i]);
            }

            // 4. HMAC-SHA512 with SK
            Mac hmac = Mac.getInstance("HmacSHA512");
            SecretKeySpec keySpec = new SecretKeySpec(SK, "HmacSHA512");
            hmac.init(keySpec);
            byte[] hmacBytes = hmac.doFinal(xored);

            // 5. Base64 인코딩 → CI (보통 88Byte 문자열)
            return Base64.getEncoder().encodeToString(hmacBytes);

        } catch (Exception e) {
            throw new RuntimeException("CI 생성 실패", e);
        }
    }

}
