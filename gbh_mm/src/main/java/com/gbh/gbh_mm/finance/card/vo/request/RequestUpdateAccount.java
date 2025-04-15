package com.gbh.gbh_mm.finance.card.vo.request;

import lombok.Data;

@Data
public class RequestUpdateAccount {
    private String userKey;
    private String cardNo;
    private String cvc;
    private String withdrawalAccountNo;
    private String withdrawalDate;
}
