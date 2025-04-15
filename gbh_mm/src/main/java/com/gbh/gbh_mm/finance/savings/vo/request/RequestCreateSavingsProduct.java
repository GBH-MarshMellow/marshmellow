package com.gbh.gbh_mm.finance.savings.vo.request;

import lombok.Data;

@Data
public class RequestCreateSavingsProduct {
    private String bankCode;
    private String accountName;
    private String accountDescription;
    private String subscriptionPeriod;
    private long maxSubscriptionBalance;
    private long minSubscriptionBalance;
    private double interestRate;
    private String rateDescription;
}
