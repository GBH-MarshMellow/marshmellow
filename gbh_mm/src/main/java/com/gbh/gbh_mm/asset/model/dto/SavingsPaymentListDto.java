package com.gbh.gbh_mm.asset.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SavingsPaymentListDto {
    private String depositInstallment;
    private String paymentBalance;
    private String paymentDate;
    private String paymentTime;
    private String status;
    private String failureReason;
}
