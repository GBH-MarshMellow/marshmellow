package com.gbh.gbh_mm.finance.demandDeposit.vo.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class RequestAccountTransfer {
    private String depositAccountNo;
    private String depositTransactionSummary;
    private long transactionBalance;
    private String withdrawalAccountNo;
    private String withdrawalTransactionSummary;
    private String userKey;
}
