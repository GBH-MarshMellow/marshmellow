package com.gbh.gbh_mm.finance.auth.vo.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RequestCheckAccountAuth {
    private String userKey;
    private String accountNo;
    private String authCode;
}
