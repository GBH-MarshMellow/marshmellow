package com.gbh.gbh_mm.budget.model.response;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Builder
public class ResponseUpdateBudgetAlarm {
    private String message;

    private LocalDateTime newAlarmTime;
}
