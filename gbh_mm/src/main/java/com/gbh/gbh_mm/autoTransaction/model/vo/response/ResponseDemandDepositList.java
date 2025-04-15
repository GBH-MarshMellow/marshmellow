package com.gbh.gbh_mm.autoTransaction.model.vo.response;

import com.gbh.gbh_mm.autoTransaction.model.dto.DemandDepositDto;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ResponseDemandDepositList {
    private List<DemandDepositDto> demandDepositList;
}
