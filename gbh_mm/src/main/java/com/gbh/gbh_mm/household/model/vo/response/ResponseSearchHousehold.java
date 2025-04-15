package com.gbh.gbh_mm.household.model.vo.response;

import com.gbh.gbh_mm.household.model.dto.DateGroupDto;
import com.gbh.gbh_mm.household.model.entity.Household;
import lombok.*;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ResponseSearchHousehold {
    List<DateGroupDto> householdList;
}
