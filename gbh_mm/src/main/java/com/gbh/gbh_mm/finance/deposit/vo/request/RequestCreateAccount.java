package com.gbh.gbh_mm.finance.deposit.vo.request;

import lombok.Data;

@Data
public class RequestCreateAccount {
    private String userKey;
    private String accountTypeUniqueNo;
    private String withdrawalAccountNo;
    private long depositBalance;
}
