package com.gbh.gbh_mm.asset.model.vo.response;

import com.gbh.gbh_mm.asset.model.dto.CardListDto;
import com.gbh.gbh_mm.asset.model.dto.DemandDepositListDto;
import com.gbh.gbh_mm.asset.model.dto.DepositListDto;
import com.gbh.gbh_mm.asset.model.dto.LoanListDto;
import com.gbh.gbh_mm.asset.model.dto.SavingsListDto;
import java.util.List;
import lombok.Data;

@Data
public class ResponseFindAssetList {
    private String iv;
    private CardListDto cardData;
    private DemandDepositListDto demandDepositData;
    private LoanListDto loanData;
    private SavingsListDto savingsData;
    private DepositListDto depositData;
}
