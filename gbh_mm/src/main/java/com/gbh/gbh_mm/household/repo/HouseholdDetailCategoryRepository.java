package com.gbh.gbh_mm.household.repo;

import com.gbh.gbh_mm.household.model.entity.HouseholdDetailCategory;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface HouseholdDetailCategoryRepository extends
    JpaRepository<HouseholdDetailCategory, Integer> {

    HouseholdDetailCategory findByHouseholdDetailCategory(String householdDetailCategoryName);
}
