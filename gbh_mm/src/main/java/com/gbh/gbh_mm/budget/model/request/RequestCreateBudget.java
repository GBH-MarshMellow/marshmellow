package com.gbh.gbh_mm.budget.model.request;

import lombok.Data;

@Data
public class RequestCreateBudget {
    private long salary;

    private float fixedExpense;

    private float foodExpense;

    private float transportationExpense;

    private float marketExpense;

    private float financialExpense;

    private float leisureExpense;

    private float coffeeExpense;

    private float shoppingExpense;

    private float emergencyExpense;


}
