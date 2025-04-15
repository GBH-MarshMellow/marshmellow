package com.gbh.gbh_mm.asset.model.entity;

import lombok.Data;

@Data
public class Loan {
    private String accountNo;
    private String accountName;
    private long loanBalance;
    private String encodeLoanBalance;
}
