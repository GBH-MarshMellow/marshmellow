package com.gbh.gbh_mm.finance.user.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.BankAPI;
import com.gbh.gbh_mm.api.UserAPI;
import com.gbh.gbh_mm.finance.user.vo.request.RequestCreateUserKey;
import com.gbh.gbh_mm.finance.user.vo.request.RequestReissueUserKey;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class UserFinServiceImpl implements UserFinService {
    private final UserAPI userAPI;

    @Override
    public Map<String, Object> createUserKey(RequestCreateUserKey request)
        throws JsonProcessingException {

        return userAPI.createUserKey(request);
    }

    @Override
    public Map<String, Object> searchUser(RequestReissueUserKey request)
        throws JsonProcessingException {
        return userAPI.searchUser(request);
    }
}
