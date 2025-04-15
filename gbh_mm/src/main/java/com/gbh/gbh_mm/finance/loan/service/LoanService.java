package com.gbh.gbh_mm.finance.loan.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccountList;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestCreateAudit;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestCreateLoanAccount;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestCreateLoanProduct;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFindAuditList;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFindRepaymentList;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFullRepayment;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestUserRating;
import java.util.Map;

public interface LoanService {

    Map<String, Object> findAssetRating()
        throws JsonProcessingException;

    Map<String, Object> createProduct(RequestCreateLoanProduct request)
        throws JsonProcessingException;

    Map<String, Object> findProductList()
        throws JsonProcessingException;

    Map<String, Object> findUserRating(RequestUserRating request)
        throws JsonProcessingException;

    Map<String, Object> createAudit(RequestCreateAudit request)
        throws JsonProcessingException;

    Map<String, Object> findAuditList(RequestFindAuditList request)
        throws JsonProcessingException;

    Map<String, Object> createAccount(RequestCreateLoanAccount request)
        throws JsonProcessingException;

    Map<String, Object> findAccountList(RequestFindAccountList request)
        throws JsonProcessingException;

    Map<String, Object> findRepaymentList(RequestFindRepaymentList request)
        throws JsonProcessingException;

    Map<String, Object> fullRepayement(RequestFullRepayment request)
        throws JsonProcessingException;
}
