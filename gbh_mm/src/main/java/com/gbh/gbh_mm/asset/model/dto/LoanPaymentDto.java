package com.gbh.gbh_mm.asset.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class LoanPaymentDto {
    private String installmentNumber;
    private String status;
    private String paymentBalance;
    private String repaymentAttemptDate;
    private String repaymentAttemptTime;
    private String repaymentActualDate;
    private String repaymentActualTime;
    private String failureReason;
}
