package com.gbh.gbh_mm.household.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class PaymentMethodDto {
    private String bankCode;
    private String bankName;
    private String paymentType;
    private String paymentMethod;
}
