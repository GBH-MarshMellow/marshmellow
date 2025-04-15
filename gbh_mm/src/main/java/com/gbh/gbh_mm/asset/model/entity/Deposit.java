package com.gbh.gbh_mm.asset.model.entity;

import lombok.Data;

@Data
public class Deposit {
    private String bankCode;
    private String bankName;
    private String accountNo;
    private String accountName;
    private long depositBalance;
    private String encodeDepositBalance;
}
