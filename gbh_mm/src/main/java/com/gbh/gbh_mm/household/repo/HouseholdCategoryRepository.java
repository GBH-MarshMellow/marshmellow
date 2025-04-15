package com.gbh.gbh_mm.household.repo;

import com.gbh.gbh_mm.household.model.entity.HouseholdCategory;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface HouseholdCategoryRepository extends JpaRepository<HouseholdCategory, Integer> {

}
