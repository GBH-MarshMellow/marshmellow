package com.gbh.gbh_mm.delusion.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.Map;

@NoArgsConstructor
@AllArgsConstructor
@Builder
@Getter
public class AverageSpendingResponseDto {

    private Map<String, Long> monthlySpendingMap;
    private Long averageMonthlySpending;

}
