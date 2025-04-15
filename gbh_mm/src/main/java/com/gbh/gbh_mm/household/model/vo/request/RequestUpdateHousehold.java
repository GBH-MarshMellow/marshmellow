package com.gbh.gbh_mm.household.model.vo.request;

import lombok.Data;

@Data
public class RequestUpdateHousehold {
    private long householdPk;
    private String householdMemo;
    private Integer householdAmount;
    private String exceptedBudgetYn;
    private int householdDetailCategoryPk;
}
