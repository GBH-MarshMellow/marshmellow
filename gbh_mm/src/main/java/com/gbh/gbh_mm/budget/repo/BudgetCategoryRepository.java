package com.gbh.gbh_mm.budget.repo;

import com.gbh.gbh_mm.budget.model.entity.BudgetCategory;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface BudgetCategoryRepository extends JpaRepository<BudgetCategory, Long> {

    // 세부 예산 조회
    List<BudgetCategory> findAllByBudget_BudgetPk(Long budgetPk);
}
