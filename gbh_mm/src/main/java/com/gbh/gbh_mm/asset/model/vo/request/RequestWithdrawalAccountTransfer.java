package com.gbh.gbh_mm.asset.model.vo.request;

import lombok.Data;

@Data
public class RequestWithdrawalAccountTransfer {
    private String iv;
    private int withdrawalAccountId;
    private String depositAccountNo;
    private String transactionSummary;
    private String transactionBalance;
}
