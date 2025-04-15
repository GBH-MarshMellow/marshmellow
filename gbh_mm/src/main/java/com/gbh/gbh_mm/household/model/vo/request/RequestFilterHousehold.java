package com.gbh.gbh_mm.household.model.vo.request;

import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import lombok.Getter;

@Getter
public class RequestFilterHousehold {
    private String startDate;
    private String endDate;
    private HouseholdClassificationEnum classification;
}
