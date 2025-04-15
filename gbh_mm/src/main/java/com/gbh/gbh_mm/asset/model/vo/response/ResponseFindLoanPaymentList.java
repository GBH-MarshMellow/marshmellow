package com.gbh.gbh_mm.asset.model.vo.response;

import com.gbh.gbh_mm.asset.model.dto.LoanPaymentDto;
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
public class ResponseFindLoanPaymentList {
    private String iv;
    private String status;
    private String loanBalance;
    private String remainingLoanBalance;
    private List<LoanPaymentDto> repaymentRecords;
}
