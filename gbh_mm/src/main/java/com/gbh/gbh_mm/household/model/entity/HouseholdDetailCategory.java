package com.gbh.gbh_mm.household.model.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "tbl_household_detail_category")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class HouseholdDetailCategory {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "household_detail_category_pk")
    private int householdDetailCategoryPk;

    @Column(name = "household_detail_category")
    private String householdDetailCategory;

    @ManyToOne
    @JoinColumn(name = "ai_category_pk")
    private AiCategory aiCategory;

    @ManyToOne
    @JoinColumn(name = "household_category_pk")
    private HouseholdCategory householdCategory;

}
