package com.gbh.gbh_mm.household.model.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.Getter;

@Entity
@Table(name = "tbl_household_category")
@Getter
public class HouseholdCategory {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "household_category_pk")
    private int householdCategoryPk;

    @Column(name = "household_category_name")
    private String householdCategoryName;

    @Column(name = "household_category_image_url")
    private String householdCategoryImageUrl;
}
