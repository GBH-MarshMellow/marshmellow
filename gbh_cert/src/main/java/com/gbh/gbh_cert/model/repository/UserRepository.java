package com.gbh.gbh_cert.model.repository;

import com.gbh.gbh_cert.model.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByConnectionInformation(String connectionInformation);

    Optional<User> findByEmail(String email);

    boolean existsByEmail(String email);
}
