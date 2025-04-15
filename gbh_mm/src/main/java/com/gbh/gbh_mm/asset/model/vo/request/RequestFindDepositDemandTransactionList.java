package com.gbh.gbh_mm.asset.model.vo.request;

import lombok.Data;

@Data
public class RequestFindDepositDemandTransactionList {
    private String iv;
    private String accountNo;
    private String startDate;
    private String endDate;
    private String transactionType;
    private String orderByType;
}
