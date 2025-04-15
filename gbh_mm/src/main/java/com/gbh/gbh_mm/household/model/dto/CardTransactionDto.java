package com.gbh.gbh_mm.household.model.dto;

import lombok.Data;

@Data
public class CardTransactionDto {
    private String merchantName;
    private String transactionDate;
    private String transactionTime;
    private long transactionBalance;
    private String cardStatus;
}
