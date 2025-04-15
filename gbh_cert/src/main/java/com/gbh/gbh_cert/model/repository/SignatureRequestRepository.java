package com.gbh.gbh_cert.model.repository;

import com.gbh.gbh_cert.model.entity.SignatureRequest;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface SignatureRequestRepository extends JpaRepository<SignatureRequest, Long> {
}
