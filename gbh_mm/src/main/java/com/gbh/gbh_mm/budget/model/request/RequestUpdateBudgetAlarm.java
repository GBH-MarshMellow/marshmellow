package com.gbh.gbh_mm.budget.model.request;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class RequestUpdateBudgetAlarm {
    private LocalDateTime budgetAlarmTime;
}
