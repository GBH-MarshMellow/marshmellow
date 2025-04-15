package com.gbh.gbh_mm.finance.card.vo.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RequestFindCardTransactionList {
    private String userKey;
    private String cardNo;
    private String cvc;
    private String startDate;
    private String endDate;
}
