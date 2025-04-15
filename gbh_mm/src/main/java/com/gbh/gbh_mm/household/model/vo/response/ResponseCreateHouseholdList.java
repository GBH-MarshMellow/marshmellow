package com.gbh.gbh_mm.household.model.vo.response;

import com.gbh.gbh_mm.household.model.dto.DateGroupDto;
import com.gbh.gbh_mm.household.model.dto.HouseHoldDto;
import com.gbh.gbh_mm.household.model.entity.Household;
import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ResponseCreateHouseholdList {
    private long totalIncome;
    private long totalExpenditure;
    List<DateGroupDto> houseHoldList;
}
