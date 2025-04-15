package com.gbh.gbh_mm.autoTransaction.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DemandDepositDto {
    private String accountNo;
    private String bankName;
}
