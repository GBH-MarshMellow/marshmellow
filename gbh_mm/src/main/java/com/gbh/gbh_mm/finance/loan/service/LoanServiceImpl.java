package com.gbh.gbh_mm.finance.loan.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.LoanAPI;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccountList;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestCreateAudit;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestCreateLoanAccount;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestCreateLoanProduct;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFindAuditList;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFindRepaymentList;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFullRepayment;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestUserRating;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class LoanServiceImpl implements LoanService {
    private final LoanAPI loanAPI;

    @Override
    public Map<String, Object> findAssetRating() throws JsonProcessingException {
        return loanAPI.findAssetRating();
    }

    @Override
    public Map<String, Object> createProduct(RequestCreateLoanProduct request)
        throws JsonProcessingException {
        return loanAPI.createProduct(request);
    }

    @Override
    public Map<String, Object> findProductList() throws JsonProcessingException {
        return loanAPI.findProductList();
    }

    @Override
    public Map<String, Object> findUserRating(RequestUserRating request)
        throws JsonProcessingException {
        return loanAPI.findUserRating(request);
    }

    @Override
    public Map<String, Object> createAudit(RequestCreateAudit request)
        throws JsonProcessingException {
        return loanAPI.createAudit(request);
    }

    @Override
    public Map<String, Object> findAuditList(RequestFindAuditList request)
        throws JsonProcessingException {
        return loanAPI.findAuditList(request);
    }

    @Override
    public Map<String, Object> createAccount(RequestCreateLoanAccount request)
        throws JsonProcessingException {
        return loanAPI.createAccount(request);
    }

    @Override
    public Map<String, Object> findAccountList(RequestFindAccountList request)
        throws JsonProcessingException {
        return loanAPI.findAccountList(request.getUserKey());
    }

    @Override
    public Map<String, Object> findRepaymentList(RequestFindRepaymentList request)
        throws JsonProcessingException {
        return loanAPI.findRepaymentList(request);
    }

    @Override
    public Map<String, Object> fullRepayement(RequestFullRepayment request)
        throws JsonProcessingException {
        return loanAPI.fullRepayment(request);
    }
}
