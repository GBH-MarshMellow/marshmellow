package com.gbh.gbh_cert.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_cert.api.UserAPI;
import com.gbh.gbh_cert.exception.CustomException;
import com.gbh.gbh_cert.exception.ErrorCode;
import com.gbh.gbh_cert.model.dto.request.CIRequestDto;
import com.gbh.gbh_cert.model.dto.request.RequestCreateUserKey;
import com.gbh.gbh_cert.model.entity.User;
import com.gbh.gbh_cert.model.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final UserAPI userAPI;

    @Transactional
    public void registerUserIfNotExist(CIRequestDto ciRequestDto, String ci) {
        userRepository.findByConnectionInformation(ci).orElseGet(() ->
                userRepository.save(User.builder()
                        .connectionInformation(ci)
                        .name(ciRequestDto.getUserName())
                        .phoneNumber(ciRequestDto.getPhoneNumber())
                        .birth(ciRequestDto.getUserCode())
                        .build())
        );
    }
    /**
     * 사용자 이메일 중복체크
     */
    @Transactional(readOnly = true)
    public boolean  isEmailExists(String email) {
        return userRepository.existsByEmail(email);
    }

    /**
     * CI(Connection Information)로 사용자 조회
     */
    @Transactional(readOnly = true)
    public User lookUpUserByCI(String connectionInformation) {

        return userRepository.findByConnectionInformation(connectionInformation)
                .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));
    }

    @Transactional
    public String updateUserKey(User user, String userEmail) throws JsonProcessingException {
        Map<String, Object> userKeyMap = userAPI.createUserKey(RequestCreateUserKey.builder()
                        .userId(userEmail)
                        .build());
        Map<String, Object> apiResponse = (Map<String, Object>) userKeyMap.get("apiResponse");
        String userKey = (String) apiResponse.get("userKey");

        int mid = userKey.length() / 2;
        String firstHalf = userKey.substring(0, mid);
        String secondHalf = userKey.substring(mid);

        user.updateHalfUserKey(secondHalf);
        userRepository.save(user);
        return firstHalf;
    }

}
