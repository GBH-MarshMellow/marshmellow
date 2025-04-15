package com.gbh.gbh_mm.budget.model.response;

import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class ResponseFindBudgetCategoryList {
    private String message;
    private List<BudgetCategoryData> budgetCategoryList;

    @Data
    @Builder
    public static class BudgetCategoryData {
        private Long budgetCategoryPk;
        private String budgetCategoryName;
        private Long budgetCategoryPrice;
        private Long budgetExpendAmount;
        private double budgetExpendPercent;
    }
}
