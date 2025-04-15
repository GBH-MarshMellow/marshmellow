package com.gbh.gbh_mm.finance.savings.vo.request;

import lombok.Data;

@Data
public class RequestCreateSavingsAccount {
    private String userKey;
    private String accountTypeUniqueNo;
    private String withdrawalAccountNo;
    private long depositBalance;
}
