package com.gbh.gbh_mm.finance.auth.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.AuthAPI;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCheckAccountAuth;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCreateAccountAuth;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
@AllArgsConstructor
public class AuthServiceImpl implements AuthService{
    private final AuthAPI authAPI;

    @Override
    public Map<String, Object> createAccountAuth(RequestCreateAccountAuth request) throws JsonProcessingException {
        return authAPI.createAccountAuth(request);
    }

    @Override
    public Map<String, Object> checkAccountAuth(RequestCheckAccountAuth request) throws JsonProcessingException {
        return authAPI.checkAccountAuth(request);
    }
}
