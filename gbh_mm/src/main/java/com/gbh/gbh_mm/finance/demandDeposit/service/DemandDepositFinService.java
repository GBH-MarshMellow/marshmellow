package com.gbh.gbh_mm.finance.demandDeposit.service;

import com.fasterxml.jackson.core.JsonProcessingException;
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

public interface DemandDepositFinService {

    Map<String, Object> findDepositList() throws JsonProcessingException;

    Map<String, Object> createDepositProduct(RequestCreateDepositProduct request)
        throws JsonProcessingException;

    Map<String, Object> createDemandDepositAccount(RequestCreateDemandDepositAccount request)
        throws JsonProcessingException;

    Map<String, Object> findDemandDespositAccountList(RequestFindDemandDepositAccountList request)
        throws JsonProcessingException;

    Map<String, Object> findDemandDespositAccount(RequestFindDemandDepositAccount request)
        throws JsonProcessingException;

    Map<String, Object> findHolderName(RequestFindHolderName request)
        throws JsonProcessingException;

    Map<String, Object> findBalance(RequestFindBalance request)
        throws JsonProcessingException;

    Map<String, Object> withdrawal(RequestDemandDepositWithdrawal request)
        throws JsonProcessingException;

    Map<String, Object> deposit(RequestDemandDepositDeposit request)
        throws JsonProcessingException;

    Map<String, Object> accountTransfer(RequestAccountTransfer request)
        throws JsonProcessingException;

    Map<String, Object> findTransactionList(RequestFindTransactionList request)
        throws JsonProcessingException;

    Map<String, Object> deleteDemandDepositAccount(RequestDeleteDemandDepositAccount request)
        throws JsonProcessingException;
}
