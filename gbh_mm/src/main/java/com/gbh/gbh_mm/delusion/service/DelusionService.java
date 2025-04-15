package com.gbh.gbh_mm.delusion.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.CardAPI;
import com.gbh.gbh_mm.api.DemandDepositAPI;
import com.gbh.gbh_mm.api.SavingsAPI;
import com.gbh.gbh_mm.asset.model.entity.*;
import com.gbh.gbh_mm.asset.model.vo.response.ResponseFindAssetList;
import com.gbh.gbh_mm.asset.service.AssetService;
import com.gbh.gbh_mm.delusion.response.AvailableAmountResponseDto;
import com.gbh.gbh_mm.delusion.response.AverageSpendingResponseDto;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindBilling;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindTransactionList;
import com.gbh.gbh_mm.household.model.entity.Household;
import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import com.gbh.gbh_mm.household.service.HouseholdService;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.modelmapper.ModelMapper;
import org.modelmapper.convention.MatchingStrategies;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Service
@AllArgsConstructor
@Slf4j
public class DelusionService {

    private final AssetService assetService;
    private final ModelMapper mapper;
    private final CardAPI cardAPI;
    private final DemandDepositAPI demandDepositAPI;
    private final HouseholdService householdService;
    private final SavingsAPI savingsAPI;

    public AvailableAmountResponseDto getAvailableAmount(CustomUserDetails customUserDetails) {

        ResponseFindAssetList findAssetList = assetService.findAssetListWithNoEncrypt(customUserDetails);
        return AvailableAmountResponseDto.builder()
                .availableAmount(findAssetList.getDemandDepositData().getTotalAmount())
                .build();
    }

    public AverageSpendingResponseDto getAverageSpending(CustomUserDetails customUserDetails) {
        Long userPk = customUserDetails.getUserPk();
        int salaryDate = customUserDetails.getSalaryDate();

        Map<String, Long> monthlySpendingMap = householdService.findMonthlyWithdrawalMap(userPk, salaryDate);

        long totalSpending = monthlySpendingMap.values().stream().mapToLong(Long::longValue).sum();
        long averageMonthlySpending = totalSpending / 12;

        return AverageSpendingResponseDto.builder()
                .monthlySpendingMap(monthlySpendingMap)
                .averageMonthlySpending(averageMonthlySpending)
                .build();
    }
}
