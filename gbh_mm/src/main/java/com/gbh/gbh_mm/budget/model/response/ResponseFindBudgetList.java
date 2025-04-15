package com.gbh.gbh_mm.budget.model.response;

import com.gbh.gbh_mm.budget.model.entity.BudgetCategory;
import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class ResponseFindBudgetList {
    private String message;
    private List<BudgetData> budgetList;

    @Data
    @Builder
    public static class BudgetData {
        private Long budgetPk;
        private Long budgetAmount;
        private String startDate;
        private String endDate;
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
}
