package com.gbh.gbh_mm.finance.demandDeposit.vo.request;

import lombok.Data;

@Data
public class RequestCreateDepositProduct {
    private String bankCode;
    private String accountName;
    private String accountDescription;

}
