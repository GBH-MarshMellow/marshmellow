package com.gbh.gbh_mm.household.model.vo.request;

import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import lombok.Data;

@Data
public class RequestCreateHousehold {
    private String tradeName;
    private String tradeDate;
    private String tradeTime;
    private String householdAmount;
    private String householdMemo;
    private String paymentMethod;
    private String exceptedBudgetYn;
    private HouseholdClassificationEnum householdClassification;
    private int householdDetailCategoryPk;
}
