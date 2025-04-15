package com.gbh.gbh_cert.model.repository;

import com.gbh.gbh_cert.model.entity.SignatureVerification;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface SignatureVerificationRepository extends JpaRepository<SignatureVerification, Long> {
}
