package com.gbh.gbh_cert.service;

import com.gbh.gbh_cert.exception.CustomException;
import com.gbh.gbh_cert.exception.ErrorCode;
import com.gbh.gbh_cert.model.entity.Organization;
import com.gbh.gbh_cert.model.repository.OrganizationRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class OrganizationService {

    private final OrganizationRepository organizationRepository;

    public Organization getValidOrganization(String orgCode) {
        return organizationRepository.findById(orgCode)
                .orElseThrow(() -> new CustomException(ErrorCode.BAD_REQUEST));
    }

    public List<Organization> getValidOrganizations(List<String> orgCodes) {
        return orgCodes.stream()
                .map(this::getValidOrganization)
                .toList();
    }
}
