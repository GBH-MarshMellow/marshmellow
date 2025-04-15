package com.gbh.gbh_mm.household.model.dto;

import com.gbh.gbh_mm.household.model.entity.HouseholdDetailCategory;
import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import com.gbh.gbh_mm.user.model.entity.User;
import jakarta.persistence.Column;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import lombok.Data;

@Data
public class HouseHoldDto {
    private long householdPk;
    private String tradeName;
    private String tradeDate;
    private String tradeTime;
    private int householdAmount;
    private String householdMemo;
    private String paymentMethod;
    private String paymentCancelYn;
    private String exceptedBudgetYn;
    private User user;
    private String category;
    private HouseholdClassificationEnum householdClassificationCategory;

}
