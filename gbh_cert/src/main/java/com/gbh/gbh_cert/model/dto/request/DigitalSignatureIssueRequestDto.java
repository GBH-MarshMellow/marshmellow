package com.gbh.gbh_cert.model.dto.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DigitalSignatureIssueRequestDto {

    private String signedData; // base64로 인코딩된 서명 결과
    private String originalText;
    private String halfUserKey;
    private String certificatePem;
    private String connectionInformation;
    private List<String> orgList;
}
