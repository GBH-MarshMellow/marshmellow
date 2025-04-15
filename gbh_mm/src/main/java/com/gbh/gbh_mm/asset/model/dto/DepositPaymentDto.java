package com.gbh.gbh_mm.asset.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DepositPaymentDto {
    String paymentUniqueNo;
    String paymentDate;
    String paymentTime;
    String paymentBalance;
}
