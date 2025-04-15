package com.gbh.gbh_mm.household.model.entity;

import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import com.gbh.gbh_mm.user.model.entity.User;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import java.sql.Time;
import java.util.Date;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "tbl_household")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Household {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "household_pk")
    private long householdPk;

    @Column(name = "trade_name")
    private String tradeName;

    @Column(name = "trade_date")
    private String tradeDate;

    @Column(name = "trade_time")
    private String tradeTime;

    @Column(name = "household_amount")
    private int householdAmount;

    @Column(name = "household_memo")
    private String householdMemo;

    @Column(name = "payment_method")
    private String paymentMethod;

    @Column(name = "payment_cancel_yn")
    private String paymentCancelYn;

    @Column(name = "excepted_budget_yn")
    private String exceptedBudgetYn;

    @ManyToOne
    @JoinColumn(name = "user_pk")
    private User user;

    @ManyToOne
    @JoinColumn(name = "household_detail_category_pk")
    private HouseholdDetailCategory householdDetailCategory;

    @Column(name = "household_classification")
    @Enumerated(EnumType.STRING)
    private HouseholdClassificationEnum householdClassificationCategory;
}
