package com.gbh.gbh_mm.user.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DepositDto {
    private String transactionDate;
    private String transactionTime;
    private long transactionBalance;
    private String transactionSummary;
    private String transactionMemo;
}
