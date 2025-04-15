package com.gbh.gbh_mm.user.service;

import com.gbh.gbh_mm.common.exception.CustomException;
import com.gbh.gbh_mm.common.exception.ErrorCode;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.repo.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
@RequiredArgsConstructor
public class CustomUserDetailService implements UserDetailsService {

    private final UserRepository userRepository;

    @Override
    public UserDetails loadUserByUsername(String userPk) {
        Optional<User> user = userRepository.findByUserPk(Long.valueOf(userPk));
        if (user.isEmpty()) {
            throw new CustomException(ErrorCode.CHILD_NOT_FOUND);
        }
        return user.map(CustomUserDetails::new)
                .orElseThrow(() -> new CustomException(ErrorCode.BAD_REQUEST));
    }
}
