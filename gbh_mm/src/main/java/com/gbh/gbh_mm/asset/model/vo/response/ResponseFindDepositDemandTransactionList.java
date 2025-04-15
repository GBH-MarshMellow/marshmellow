package com.gbh.gbh_mm.asset.model.vo.response;

import com.gbh.gbh_mm.asset.model.dto.DemandDepositTransactionDto;
import lombok.Builder;
import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class ResponseFindDepositDemandTransactionList {
    String iv;
    List<DemandDepositTransactionDto> transactionList;
}
