package com.gbh.gbh_cert.model.repository;

import com.gbh.gbh_cert.model.entity.Certificate;
import com.gbh.gbh_cert.model.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface CertficateRepository extends JpaRepository<Certificate, Long> {
    Optional<Certificate> findByUser(User user);

    Optional<Certificate> findByUserAndCertStatus(User user, Certificate.CertStatus certStatus);
}
