package com.gbh.gbh_mm.user.repo;

import com.gbh.gbh_mm.user.model.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    User findByUserKey(String userKey);

    boolean existsByPhoneNumber(String phoneNumber);

    Optional<User> findByUserPk(Long userPk);

    Optional<User> findByPhoneNumber(String phoneNumber);

    List<User> findBySalaryDate(int today);
}
