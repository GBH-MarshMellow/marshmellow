package com.gbh.gbh_mm.household.service;

import com.gbh.gbh_mm.household.model.entity.Household;
import com.gbh.gbh_mm.household.model.vo.request.*;
import com.gbh.gbh_mm.household.model.vo.response.*;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;

import java.util.List;
import java.util.Map;

public interface HouseholdService {

    ResponseFindHouseholdList findHouseholdList(RequestFindHouseholdList request,
        CustomUserDetails customUserDetails);

    ResponseCreateHousehold createHousehold(RequestCreateHousehold request,
        CustomUserDetails customUserDetails);

    ResponseFindHousehold findHousehold(RequestFindHousehold request);

    ResponseUpdateHousehold updateHousehold(RequestUpdateHousehold request);

    ResponseDeleteHousehold deleteHousehold(RequestDeleteHousehold request);

    ResponseFindTransactionDataList findTransactionDataList(CustomUserDetails customUserDetails);

    ResponseCreateHouseholdList createHouseholdList(RequestCreateHouseholdList request);

    ResponseSearchHousehold searchHousehold(RequestSearchHousehold request,
        CustomUserDetails customUserDetails);

    ResponseFilterHousehold filterHousehold(RequestFilterHousehold request,
        CustomUserDetails customUserDetails);

    ResponsePaymentMethodList findPaymentMethodList(CustomUserDetails customUserDetails);

    ResponseAiAvg findAiAvg(CustomUserDetails customUserDetails);

    public Map<String, Long> findMonthlyWithdrawalMap(Long userPk, int salaryDate);
}
