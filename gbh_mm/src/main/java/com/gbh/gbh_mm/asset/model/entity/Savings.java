package com.gbh.gbh_mm.asset.model.entity;

import lombok.Data;

@Data
public class Savings {
    private String bankCode;
    private String bankName;
    private String accountNo;
    private String accountName;
    private long totalBalance;
    private String encodedTotalBalance;
    private String subscriptionPeriod;
    private String installmentNumber;
    private String depositBalance;
    private String accountExpiryDate;
}
