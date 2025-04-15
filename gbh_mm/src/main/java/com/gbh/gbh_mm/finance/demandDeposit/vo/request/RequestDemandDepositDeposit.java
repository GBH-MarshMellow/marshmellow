package com.gbh.gbh_mm.finance.demandDeposit.vo.request;

import lombok.Data;

@Data
public class RequestDemandDepositDeposit {
    private String accountNo;
    private long transactionBalance;
    private String transactionSummary;
    private String userKey;
}
