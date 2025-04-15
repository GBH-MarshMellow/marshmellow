package com.gbh.gbh_mm.finance.user.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.user.vo.request.RequestCreateUserKey;
import com.gbh.gbh_mm.finance.user.vo.request.RequestReissueUserKey;
import java.util.Map;

public interface UserFinService {

    Map<String, Object> createUserKey(RequestCreateUserKey request) throws JsonProcessingException;

    Map<String, Object> searchUser(RequestReissueUserKey request) throws JsonProcessingException;
}
