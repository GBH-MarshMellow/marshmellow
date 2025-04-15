package com.gbh.gbh_mm.finance.savings.vo.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RequestFindSavingsPayment {
    private String userKey;
    private String accountNo;
}
