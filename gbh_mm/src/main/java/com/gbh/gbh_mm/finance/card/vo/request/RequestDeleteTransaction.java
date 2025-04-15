package com.gbh.gbh_mm.finance.card.vo.request;

import lombok.Data;

@Data
public class RequestDeleteTransaction {
    private String userKey;
    private String cardNo;
    private String cvc;
    private long transactionUniqueNo;

}
