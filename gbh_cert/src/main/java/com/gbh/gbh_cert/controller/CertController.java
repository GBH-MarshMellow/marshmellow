package com.gbh.gbh_cert.controller;

import com.gbh.gbh_cert.model.dto.request.CIRequestDto;
import com.gbh.gbh_cert.model.dto.request.CertExistRequestDto;
import com.gbh.gbh_cert.model.dto.request.CertIssueRequestDto;
import com.gbh.gbh_cert.model.dto.request.DigitalSignatureIssueRequestDto;
import com.gbh.gbh_cert.model.dto.response.CIResponseDto;
import com.gbh.gbh_cert.model.dto.response.CertExistResponseDto;
import com.gbh.gbh_cert.model.dto.response.CertResponseDto;
import com.gbh.gbh_cert.model.dto.response.DigitalSignatureIssueResponseDto;
import com.gbh.gbh_cert.service.CertService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RequiredArgsConstructor
@RestController
@RequestMapping("/api/cert")
public class CertController {

    private final CertService certService;

    @PostMapping("/ci")
    public CIResponseDto generateCi(@Valid @RequestBody CIRequestDto ciRequestDto) {
        return certService.getConnectionInformation(ciRequestDto);
    }

    @PostMapping("/exist")
    public CertExistResponseDto isExistCertificate(@RequestBody CertExistRequestDto certExistRequestDto){
        return certService.checkCertificateExistence(certExistRequestDto);
    }

    @PostMapping("/issue")
    public CertResponseDto issueCertificate(@RequestBody CertIssueRequestDto certIssueRequestDto) throws Exception {
        return certService.createCertificate(certIssueRequestDto);
    }

    @PostMapping("/digital-signature")
    public DigitalSignatureIssueResponseDto issueDigitalSignature(@RequestBody DigitalSignatureIssueRequestDto digitalSignatureIssueRequestDto) throws Exception {
        return certService.verifyDigitalSignature(digitalSignatureIssueRequestDto);
    }

}
