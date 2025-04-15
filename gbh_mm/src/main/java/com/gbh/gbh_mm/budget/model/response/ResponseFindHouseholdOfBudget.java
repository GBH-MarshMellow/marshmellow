package com.gbh.gbh_mm.budget.model.response;

import com.gbh.gbh_mm.household.model.entity.Household;
import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class ResponseFindHouseholdOfBudget {
    private String message;

    private int totalNumberOfHouseholds;

    private long totalAmount;

    private List<Household> households;

}
