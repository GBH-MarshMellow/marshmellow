package com.gbh.gbh_mm.finance.card.vo.request;

import lombok.Data;

@Data
public class RequestCreateUserCard {
    private String userKey;
    private String cardUniqueNo;
    private String withdrawalAccountNo;
    private String withdrawalDate;
}
