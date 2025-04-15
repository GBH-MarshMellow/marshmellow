package com.gbh.gbh_mm.household.controller;

import com.gbh.gbh_mm.household.model.vo.request.*;
import com.gbh.gbh_mm.household.model.vo.response.*;
import com.gbh.gbh_mm.household.service.HouseholdService;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import lombok.AllArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/household")
@AllArgsConstructor
public class HouseholdController {

    private final HouseholdService householdService;

    @GetMapping("/list")
    public ResponseFindHouseholdList findHouseholdList(
            @RequestBody RequestFindHouseholdList request,
            @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseFindHouseholdList response = householdService
                .findHouseholdList(request, customUserDetails);

        return response;
    }

    @PostMapping
    public ResponseCreateHousehold createHousehold(
            @RequestBody RequestCreateHousehold request,
            @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseCreateHousehold response = householdService
                .createHousehold(request, customUserDetails);

        return response;
    }

    @GetMapping
    public ResponseFindHousehold findHousehold(
            @RequestBody RequestFindHousehold request
    ) {
        ResponseFindHousehold response = householdService.findHousehold(request);

        return response;
    }

    @PatchMapping
    public ResponseUpdateHousehold updateHousehold(
            @RequestBody RequestUpdateHousehold request
    ) {
        ResponseUpdateHousehold response = householdService.updateHousehold(request);

        return response;
    }

    @DeleteMapping
    public ResponseDeleteHousehold deleteHousehold(
            @RequestBody RequestDeleteHousehold request
    ) {
        ResponseDeleteHousehold response = householdService.deleteHousehold(request);

        return response;
    }

    @GetMapping("/transaction-data")
    public ResponseFindTransactionDataList findTransactionDataList(
            @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseFindTransactionDataList response = householdService
                .findTransactionDataList(customUserDetails);

        return response;
    }

    @PostMapping("/household-list")
    public ResponseCreateHouseholdList createHouseholdList(
            @RequestBody RequestCreateHouseholdList request
    ) {
        ResponseCreateHouseholdList response = householdService.createHouseholdList(request);

        return response;
    }

    @GetMapping("/search")
    public ResponseSearchHousehold searchHousehold(
            @RequestBody RequestSearchHousehold request,
            @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseSearchHousehold response = householdService
                .searchHousehold(request, customUserDetails);

        return response;
    }

    @GetMapping("/filter")
    public ResponseFilterHousehold filterHousehold(
            @RequestBody RequestFilterHousehold request,
            @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseFilterHousehold response = householdService
                .filterHousehold(request, customUserDetails);

        return response;
    }

    @GetMapping("/payment-method")
    public ResponsePaymentMethodList findPaymentMethodList(
            @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        return householdService.findPaymentMethodList(customUserDetails);
    }

    @GetMapping("/ai-avg")
    public ResponseAiAvg findAiAvg(
            @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        return householdService.findAiAvg(customUserDetails);
    }
}
