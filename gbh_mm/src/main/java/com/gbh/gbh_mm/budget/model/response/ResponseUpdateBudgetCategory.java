package com.gbh.gbh_mm.budget.model.response;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResponseUpdateBudgetCategory {
    private String message;
    private Long budgetCategoryPk;

    private Long oldBudgetCategoryPrice;
    private Long newBudgetCategoryPrice;

    private Long oldBudgetAmount;
    private Long newBudgetAmount;

}
