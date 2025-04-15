package com.gbh.gbh_mm.finance.deposit.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.DemandDepositAPI;
import com.gbh.gbh_mm.api.DepositAPI;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestCreateAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestCreateDepositProduct;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestDeleteAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestEarlyInterest;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccountList;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindInterest;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindPayment;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class DepositServiceImpl implements DepositService{
    private final DepositAPI depositAPI;

    @Override
    public Map<String, Object> createDepositProduct(RequestCreateDepositProduct request)
        throws JsonProcessingException {
        return depositAPI.createDepositProduct(request);
    }

    @Override
    public Map<String, Object> findProductList() throws JsonProcessingException {
        return depositAPI.findProductList();
    }

    @Override
    public Map<String, Object> createAccount(RequestCreateAccount request)
        throws JsonProcessingException {
        return depositAPI.createAccount(request);
    }

    @Override
    public Map<String, Object> findAccountList(RequestFindAccountList request)
        throws JsonProcessingException {
        return depositAPI.findAccountList(request.getUserKey());
    }

    @Override
    public Map<String, Object> findAccount(RequestFindAccount request)
        throws JsonProcessingException {
        return depositAPI.findAccount(request);
    }

    @Override
    public Map<String, Object> findPayment(RequestFindPayment request)
        throws JsonProcessingException {
        return depositAPI.findPayment(request);
    }

    @Override
    public Map<String, Object> findInterest(RequestFindInterest request)
        throws JsonProcessingException {
        return depositAPI.findInterest(request);
    }

    @Override
    public Map<String, Object> findEarlyInterest(RequestEarlyInterest request)
        throws JsonProcessingException {
        return depositAPI.findEarlyInterest(request);
    }

    @Override
    public Map<String, Object> deleteAccount(RequestDeleteAccount request)
        throws JsonProcessingException {
        return depositAPI.deleteAccount(request);
    }
}
