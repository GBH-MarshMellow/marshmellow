package com.gbh.gbh_mm.asset.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CardTransactionDto {
    private String transactionUniqueNo;
    private String merchantId;
    private String billStatementsYn;
    private String transactionBalance;
    private String transactionDate;
    private String transactionTime;
    private String categoryName;
    private String categoryId;
    private String cardStatus;
    private String merchantName;
}
