package com.gbh.gbh_mm.household.model.vo.response;

import com.gbh.gbh_mm.household.model.entity.Household;
import java.util.List;
import lombok.Data;

@Data
public class ResponseFindTransactionDataList {
    List<Household> householdList;
}
