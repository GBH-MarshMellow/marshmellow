package com.gbh.gbh_mm.finance.savings.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestCreateSavingsAccount;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestCreateSavingsProduct;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestDeleteSavingsAccount;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsAccount;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsAccountList;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsInterest;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsPayment;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestSavingsEalryInterest;
import java.util.Map;

public interface SavingsService {

    Map<String, Object> createProduct(RequestCreateSavingsProduct request)
        throws JsonProcessingException;

    Map<String, Object> findProductList()
        throws JsonProcessingException;

    Map<String, Object> createAccount(RequestCreateSavingsAccount request)
        throws JsonProcessingException;

    Map<String, Object> findAccountList(RequestFindSavingsAccountList request)
        throws JsonProcessingException;

    Map<String, Object> findAccount(RequestFindSavingsAccount request)
        throws JsonProcessingException;

    Map<String, Object> findPayment(RequestFindSavingsPayment request)
        throws JsonProcessingException;

    Map<String, Object> findInterest(RequestFindSavingsInterest request)
        throws JsonProcessingException;

    Map<String, Object> findEarlyInterest(RequestSavingsEalryInterest request)
        throws JsonProcessingException;

    Map<String, Object> deleteAccount(RequestDeleteSavingsAccount request)
        throws JsonProcessingException;
}
