package com.gbh.gbh_mm.finance.deposit.vo.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RequestFindPayment {
    private String userKey;
    private String accountNo;
}
