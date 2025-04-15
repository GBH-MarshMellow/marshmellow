package com.gbh.gbh_mm.asset.model.vo.request;

import lombok.Getter;

@Getter
public class RequestFindCardTransactionList {
    private String iv;
    private String cardNo;
    private String cvc;
    private String startDate;
    private String endDate;
}
