package com.gbh.gbh_mm.finance.auth.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCheckAccountAuth;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCreateAccountAuth;

import java.util.Map;

public interface AuthService {
    Map<String, Object> createAccountAuth(RequestCreateAccountAuth request) throws JsonProcessingException;

    Map<String, Object> checkAccountAuth(RequestCheckAccountAuth request) throws JsonProcessingException;
}
