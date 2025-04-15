package com.gbh.gbh_mm.household.model.vo.response;

import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import lombok.Data;

@Data
public class ResponseUpdateHousehold {
    private long householdPk;
    private String tradeName;
    private String tradeDate;
    private String tradeTime;
    private int householdAmount;
    private String householdMemo;
    private String paymentMethod;
    private String paymentCancelYn;
    private String exceptedBudgetYn;
    private String householdCategory;
    private String householdDetailCategory;
    private HouseholdClassificationEnum householdClassificationCategory;
}
