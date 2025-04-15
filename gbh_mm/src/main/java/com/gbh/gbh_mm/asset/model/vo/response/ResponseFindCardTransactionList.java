package com.gbh.gbh_mm.asset.model.vo.response;

import com.gbh.gbh_mm.asset.model.dto.CardTransactionDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.util.List;
import java.util.Map;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ResponseFindCardTransactionList {
    private String iv;
    private String estimatedBalance;
    private List<CardTransactionDto> transactionList;
}
