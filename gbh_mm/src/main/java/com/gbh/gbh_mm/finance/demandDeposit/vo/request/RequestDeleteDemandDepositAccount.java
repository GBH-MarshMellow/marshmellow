package com.gbh.gbh_mm.finance.demandDeposit.vo.request;

import lombok.Data;

@Data
public class RequestDeleteDemandDepositAccount {
    private String accountNo;
    private String refundAccountNo;
    private String userKey;
}
