package com.gbh.gbh_mm.budget.model.request;

import lombok.Data;

@Data
public class RequestFindHouseholdOfBudget {
    private String startDate;

    private String endDate;

    private String aiCategory;
}
