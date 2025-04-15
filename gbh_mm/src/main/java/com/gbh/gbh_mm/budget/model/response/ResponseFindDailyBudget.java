package com.gbh.gbh_mm.budget.model.response;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResponseFindDailyBudget {
    private String message;

    private Long budgetPk;

    private Long budgetAmount;

    private Long remainBudgetAmount;

    private Long dailyBudgetAmount;


}
