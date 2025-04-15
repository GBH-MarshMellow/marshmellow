package com.gbh.gbh_mm.budget.repo;

import com.gbh.gbh_mm.budget.model.entity.Budget;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface BudgetRepository extends JpaRepository<Budget, Long> {
    List<Budget> findAllByUser_UserPkOrderByBudgetPkDesc(Long userPk);
}
