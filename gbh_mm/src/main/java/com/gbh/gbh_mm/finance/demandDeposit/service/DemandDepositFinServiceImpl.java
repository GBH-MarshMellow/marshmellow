package com.gbh.gbh_mm.finance.demandDeposit.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.DemandDepositAPI;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestAccountTransfer;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestCreateDemandDepositAccount;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestCreateDepositProduct;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestDeleteDemandDepositAccount;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestDemandDepositDeposit;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestDemandDepositWithdrawal;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindBalance;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindDemandDepositAccount;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindDemandDepositAccountList;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindHolderName;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindTransactionList;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class DemandDepositFinServiceImpl implements DemandDepositFinService {
    private final DemandDepositAPI demandDepositAPI;

    @Autowired
    public DemandDepositFinServiceImpl(DemandDepositAPI demandDepositAPI) {
        this.demandDepositAPI = demandDepositAPI;
    }

    @Override
    public Map<String, Object> findDepositList() throws JsonProcessingException {
        Map<String, Object> result = demandDepositAPI.findDepositList();

        return result;
    }

    @Override
    public Map<String, Object> createDepositProduct(RequestCreateDepositProduct request)
        throws JsonProcessingException {
        return demandDepositAPI.createDepositProduct(request);
    }

    @Override
    public Map<String, Object> createDemandDepositAccount(
        RequestCreateDemandDepositAccount request) throws JsonProcessingException {
        return demandDepositAPI.createDemandDepositAccount(request);
    }

    @Override
    public Map<String, Object> findDemandDespositAccountList(
        RequestFindDemandDepositAccountList request
    ) throws JsonProcessingException {
        return demandDepositAPI.findDemandDepositAccountList(request.getUserKey());
    }

    @Override
    public Map<String, Object> findDemandDespositAccount(RequestFindDemandDepositAccount request)
        throws JsonProcessingException {
        return demandDepositAPI.findDemandDepositAccount(request);
    }

    @Override
    public Map<String, Object> findHolderName(RequestFindHolderName request)
        throws JsonProcessingException {
        return demandDepositAPI.findHolderName(request);
    }

    @Override
    public Map<String, Object> findBalance(RequestFindBalance request)
        throws JsonProcessingException {
        return demandDepositAPI.findBalance(request);
    }

    @Override
    public Map<String, Object> withdrawal(RequestDemandDepositWithdrawal request)
        throws JsonProcessingException {
        return demandDepositAPI.withdrawal(request);
    }

    @Override
    public Map<String, Object> deposit(RequestDemandDepositDeposit request)
        throws JsonProcessingException {
        return demandDepositAPI.deposit(request);
    }

    @Override
    public Map<String, Object> accountTransfer(RequestAccountTransfer request)
        throws JsonProcessingException {
        return demandDepositAPI.accountTransfer(request);
    }

    @Override
    public Map<String, Object> findTransactionList(RequestFindTransactionList request)
        throws JsonProcessingException {
        return demandDepositAPI.findTransactionList(request);
    }

    @Override
    public Map<String, Object> deleteDemandDepositAccount(RequestDeleteDemandDepositAccount request)
        throws JsonProcessingException {
        return demandDepositAPI.deleteDemandDepositAccount(request);
    }
}
