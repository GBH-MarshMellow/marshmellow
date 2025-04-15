package com.gbh.gbh_mm.finance.savings.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.SavingsAPI;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestCreateSavingsAccount;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestCreateSavingsProduct;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestDeleteSavingsAccount;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsAccount;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsAccountList;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsInterest;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsPayment;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestSavingsEalryInterest;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class SavingsServiceImpl implements SavingsService {
    private final SavingsAPI savingsAPI;

    @Override
    public Map<String, Object> createProduct(RequestCreateSavingsProduct request)
        throws JsonProcessingException {
        return savingsAPI.createProduct(request);
    }

    @Override
    public Map<String, Object> findProductList() throws JsonProcessingException {
        return savingsAPI.findProductList();
    }

    @Override
    public Map<String, Object> createAccount(RequestCreateSavingsAccount request)
        throws JsonProcessingException {
        return savingsAPI.createAccount(request);
    }

    @Override
    public Map<String, Object> findAccountList(RequestFindSavingsAccountList request)
        throws JsonProcessingException {
        return savingsAPI.findAccountList(request.getUserKey());
    }

    @Override
    public Map<String, Object> findAccount(RequestFindSavingsAccount request)
        throws JsonProcessingException {
        return savingsAPI.findAccount(request);
    }

    @Override
    public Map<String, Object> findPayment(RequestFindSavingsPayment request)
        throws JsonProcessingException {
        return savingsAPI.findPayment(request);
    }

    @Override
    public Map<String, Object> findInterest(RequestFindSavingsInterest request)
        throws JsonProcessingException {
        return savingsAPI.findInterest(request);
    }

    @Override
    public Map<String, Object> findEarlyInterest(RequestSavingsEalryInterest request)
        throws JsonProcessingException {
        return savingsAPI.findEarlyInterest(request);
    }

    @Override
    public Map<String, Object> deleteAccount(RequestDeleteSavingsAccount request)
        throws JsonProcessingException {
        return savingsAPI.deleteAccount(request);
    }
}
