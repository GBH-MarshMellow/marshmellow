package com.gbh.gbh_cert.service;

import com.gbh.gbh_cert.model.entity.SignatureRequest;
import com.gbh.gbh_cert.model.entity.SignatureVerification;
import com.gbh.gbh_cert.model.repository.SignatureVerificationRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

@Service
@RequiredArgsConstructor
public class SignatureVerificationService {

    private final SignatureVerificationRepository signatureVerificationRepository;

    @Transactional
    public void issueRequest(SignatureRequest req) {
        signatureVerificationRepository.save(SignatureVerification.builder()
                .signatureRequest(req)
                .signatureVerified(true)
                .verifiedAt(LocalDateTime.now())
                .build());
    }
}
