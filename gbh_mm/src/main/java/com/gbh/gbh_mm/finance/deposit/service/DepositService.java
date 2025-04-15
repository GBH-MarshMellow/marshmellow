package com.gbh.gbh_mm.finance.deposit.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestCreateAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestCreateDepositProduct;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestDeleteAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestEarlyInterest;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccountList;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindInterest;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindPayment;
import java.util.Map;

public interface DepositService {

    Map<String, Object> createDepositProduct(RequestCreateDepositProduct request)
        throws JsonProcessingException;

    Map<String, Object> findProductList()
        throws JsonProcessingException;

    Map<String, Object> createAccount(RequestCreateAccount request)
        throws JsonProcessingException;

    Map<String, Object> findAccountList(RequestFindAccountList request)
        throws JsonProcessingException;

    Map<String, Object> findAccount(RequestFindAccount request)
        throws JsonProcessingException;

    Map<String, Object> findPayment(RequestFindPayment request)
        throws JsonProcessingException;

    Map<String, Object> findInterest(RequestFindInterest request)
        throws JsonProcessingException;

    Map<String, Object> findEarlyInterest(RequestEarlyInterest request)
        throws JsonProcessingException;

    Map<String, Object> deleteAccount(RequestDeleteAccount request)
        throws JsonProcessingException;
}
