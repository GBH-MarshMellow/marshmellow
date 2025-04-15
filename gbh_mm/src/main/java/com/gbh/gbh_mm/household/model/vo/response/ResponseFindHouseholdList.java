package com.gbh.gbh_mm.household.model.vo.response;

import com.gbh.gbh_mm.household.model.dto.DateGroupDto;
import com.gbh.gbh_mm.household.model.entity.Household;
import java.util.List;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResponseFindHouseholdList {
    private long totalExpenditure;
    private long totalIncome;
    private List<DateGroupDto> householdList;
}
