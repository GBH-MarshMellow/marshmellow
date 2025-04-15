package com.gbh.gbh_mm.user.service;

import com.gbh.gbh_mm.api.CertAPI;
import com.gbh.gbh_mm.user.model.request.CIRequestDto;
import com.gbh.gbh_mm.user.model.request.CertExistRequestDto;
import com.gbh.gbh_mm.user.model.request.CertIssueRequestDto;
import com.gbh.gbh_mm.user.model.request.DigitalSignatureIssueRequestDto;
import com.gbh.gbh_mm.user.model.response.CIResponseDto;
import com.gbh.gbh_mm.user.model.response.CertExistResponseDto;
import com.gbh.gbh_mm.user.model.response.CertResponseDto;
import com.gbh.gbh_mm.user.model.response.DigitalSignatureIssueResponseDto;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@Slf4j
@RequiredArgsConstructor
public class CertService {

    private final CertAPI certAPI;

    public CIResponseDto createConnectionInformation(CIRequestDto ciRequestDto) {
        return certAPI.createConnectionInformation(ciRequestDto);
    }


    public CertResponseDto createCertificate(CertIssueRequestDto certIssueRequestDto) {
        return certAPI.createCertificate(certIssueRequestDto);
    }

    public CertExistResponseDto checkCertificateExistence(CertExistRequestDto certExistRequestDto) {
        return certAPI.checkCertificateExistence(certExistRequestDto);
    }

    public DigitalSignatureIssueResponseDto issueDigitalSignature(DigitalSignatureIssueRequestDto digitalSignatureIssueRequestDto) {
        return certAPI.createDigitalSignature(digitalSignatureIssueRequestDto);
    }
}
