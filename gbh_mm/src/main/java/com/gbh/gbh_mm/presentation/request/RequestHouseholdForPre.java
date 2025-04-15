package com.gbh.gbh_mm.presentation.request;

import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import lombok.Getter;

@Getter
public class RequestHouseholdForPre {
    private long userPk;
    private String tradeName;
    private String tradeDate;
    private String tradeTime;
    private int householdAmount;
    private String householdMemo;
    private String paymentMethod;
    private int householdDetailCategoryPk;
}
