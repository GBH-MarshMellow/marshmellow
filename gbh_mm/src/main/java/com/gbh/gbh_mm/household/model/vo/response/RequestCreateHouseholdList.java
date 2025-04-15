package com.gbh.gbh_mm.household.model.vo.response;

import com.gbh.gbh_mm.household.model.dto.HouseHoldDto;
import com.gbh.gbh_mm.household.model.entity.Household;
import java.util.List;
import lombok.Data;

@Data
public class RequestCreateHouseholdList {
    private List<HouseHoldDto> transactionList;
}
