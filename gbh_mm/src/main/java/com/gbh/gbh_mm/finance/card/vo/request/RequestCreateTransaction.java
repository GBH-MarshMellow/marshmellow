package com.gbh.gbh_mm.finance.card.vo.request;

import lombok.Data;

@Data
public class RequestCreateTransaction {
    private String userKey;
    private String cardNo;
    private String cvc;
    private long merchantId;
    private long paymentBalance;
}
