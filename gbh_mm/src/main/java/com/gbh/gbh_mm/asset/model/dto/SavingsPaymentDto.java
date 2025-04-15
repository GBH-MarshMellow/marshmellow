package com.gbh.gbh_mm.asset.model.dto;

import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SavingsPaymentDto {
    private String bankCode;
    private String bankName;
    private String accountNo;
    private String interestRate;
    private String depositBalance;
    private String totalBalance;
    private String accountCreateDate;
    private String accountExpiryDate;
    private List<SavingsPaymentListDto> paymentInfo;
}
