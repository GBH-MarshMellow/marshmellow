package com.gbh.gbh_mm.household.model.vo.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ResponseAiAvg {
    private long salary;
    private double fixedAvg;
    private double foodAvg;
    private double trafficAvg;
    private double martAvg;
    private double bankAvg;
    private double leisureAvg;
    private double coffeeAvg;
    private double shoppingAvg;
    private double emergencyAvg;
}
