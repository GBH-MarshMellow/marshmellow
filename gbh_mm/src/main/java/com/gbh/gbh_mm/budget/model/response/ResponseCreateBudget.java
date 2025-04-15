package com.gbh.gbh_mm.budget.model.response;

import com.gbh.gbh_mm.budget.model.entity.Budget;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResponseCreateBudget {
    private String message;
    private Long budgetPk;
    private Long budgetAmount;
    private String startDate;
    private String endDate;

}
