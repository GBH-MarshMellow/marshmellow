package com.gbh.gbh_mm.finance.loan.vo.request;

import lombok.Data;

@Data
public class RequestCreateLoanAccount {
    private String userKey;
    private String accountTypeUniqueNo;
    private String loanBalance;
    private String withdrawalAccountNo;
}
