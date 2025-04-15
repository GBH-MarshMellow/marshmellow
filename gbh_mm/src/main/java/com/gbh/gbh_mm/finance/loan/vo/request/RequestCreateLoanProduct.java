package com.gbh.gbh_mm.finance.loan.vo.request;

import lombok.Data;

@Data
public class RequestCreateLoanProduct {
    private String bankCode;
    private String accountName;
    private String accountDescription;
    private String ratingUniqueNo;
    private int loanPeriod;
    private long minLoanBalance;
    private long maxLoanBalance;
    private double interestRate;
}
