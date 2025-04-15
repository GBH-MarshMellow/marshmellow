package com.gbh.gbh_mm.user.model.response;

import java.time.LocalDateTime;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ResponseUserDetail {
    private String salaryAccount;
    private Long salaryAmount;
    private Integer salaryDate;
    private Boolean budgetFeature;
    private LocalDateTime budgetAlarmTime;
    private String userKeyYn;
}
