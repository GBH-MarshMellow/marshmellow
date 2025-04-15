package com.gbh.gbh_cert.service;

import com.gbh.gbh_cert.model.dto.request.DigitalSignatureIssueRequestDto;
import com.gbh.gbh_cert.model.entity.*;
import com.gbh.gbh_cert.model.repository.SignatureRequestRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class SignatureRequestService {

    private final SignatureRequestRepository signatureRequestRepository;
    private final OrganizationService organizationService;

    public List<SignatureRequest> storeSignatureRequest(User user, Certificate cert, DigitalSignatureIssueRequestDto dto, List<Organization> organizations) {
        List<Organization> orgs = organizationService.getValidOrganizations(dto.getOrgList());

        List<SignatureRequest> savedList = new ArrayList<>();

        for (Organization org : organizations) {
            SignatureRequest request = SignatureRequest.builder()
                    .user(user)
                    .certificate(cert)
                    .organization(org)
                    .originalText(dto.getOriginalText())
                    .signatureData(dto.getSignedData())
                    .nonce(UUID.randomUUID().toString())
                    .requestedAt(LocalDateTime.now())
                    .build();

            savedList.add(signatureRequestRepository.save(request));
        }

        return savedList;
    }
}
