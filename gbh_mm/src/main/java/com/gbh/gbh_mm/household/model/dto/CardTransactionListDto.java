package com.gbh.gbh_mm.household.model.dto;

import java.util.List;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class CardTransactionListDto {
    private CardDto card;
    private List<CardTransactionDto> cardTransaction;
}
