package com.gbh.gbh_mm.finance.card.vo.request;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class RequestFindBilling {
    private String userKey;
    private String cardNo;
    private String cvc;
    private String startMonth;
    private String endMonth;
}
