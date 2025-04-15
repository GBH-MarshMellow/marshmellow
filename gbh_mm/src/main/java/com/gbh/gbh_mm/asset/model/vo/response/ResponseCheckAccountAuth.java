package com.gbh.gbh_mm.asset.model.vo.response;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResponseCheckAccountAuth {
    private String status;
    private int withdrawalAccountId;
}
