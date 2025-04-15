package com.gbh.gbh_mm.budget.model.response;

import com.gbh.gbh_mm.budget.model.entity.BudgetCategory;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResponseCreateBudgetCategory {
    private String message;
    private Long budgetCategoryPk;
    private String budgetCategoryName;
    private Long budgetCategoryPrice;
    private Long relatedBudgetPk;

}
