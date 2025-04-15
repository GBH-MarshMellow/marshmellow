package com.gbh.gbh_mm.household.model.dto;

import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class HouseholdDetailDto {
    private long householdPk;
    private String tradeName;
    private String tradeDate;
    private String tradeTime;
    private int householdAmount;
    private String paymentMethod;
    private String paymentCancelYn;
    private String householdCategory;
    private HouseholdClassificationEnum householdClassificationCategory; // 필요시 householdClassificationCategory 이름도 포함
}
