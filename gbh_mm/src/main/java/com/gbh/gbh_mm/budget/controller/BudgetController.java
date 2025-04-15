package com.gbh.gbh_mm.budget.controller;

import com.gbh.gbh_mm.budget.model.entity.BudgetCategory;
import com.gbh.gbh_mm.budget.model.request.RequestCreateBudget;
import com.gbh.gbh_mm.budget.model.request.RequestFindHouseholdOfBudget;
import com.gbh.gbh_mm.budget.model.request.RequestUpdateBudgetAlarm;
import com.gbh.gbh_mm.budget.model.request.RequestUpdateBudgetCategory;
import com.gbh.gbh_mm.budget.model.response.*;
import com.gbh.gbh_mm.budget.service.BudgetService;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("mm/budget")
@RequiredArgsConstructor
public class BudgetController {

    private final BudgetService budgetService;

    // 예산 리스트 조회
    @GetMapping
    public ResponseFindBudgetList getBudgetList(@AuthenticationPrincipal CustomUserDetails userDetails) {
        return budgetService.getBudgetList(userDetails.getUserPk());
    }

    // 예산 생성
    @PostMapping
    public ResponseCreateBudget createBudget(@AuthenticationPrincipal CustomUserDetails userDetails, @RequestBody RequestCreateBudget requestCreateBudget) {
        return budgetService.createBudget(userDetails.getUserPk(), requestCreateBudget);
    }

    // 세부 예산 리스트 조회
    @GetMapping("/detail/{budgetPk}")
    public ResponseFindBudgetCategoryList getBudgetCategoryList(@PathVariable Long budgetPk) {
        return budgetService.getBudgetCategoryList(budgetPk);

    }

    // 세부 예산 생성
    @PostMapping("/detail/{budgetPk}")
    public ResponseCreateBudgetCategory createBudgetCategory(@PathVariable Long budgetPk, @RequestBody BudgetCategory budgetCategory) {
        return budgetService.createBudgetCategory(budgetPk, budgetCategory);

    }

    // 세부 예산 수정
    @PutMapping("/detail/{budgetCategoryPk}")
    public ResponseUpdateBudgetCategory updateBudgetCategory(@PathVariable Long budgetCategoryPk, @RequestBody RequestUpdateBudgetCategory requestUpdateBudgetCategory) {
            return budgetService.updateBudgetCategory(budgetCategoryPk, requestUpdateBudgetCategory);
    }

    // 오늘의 예산 조회
    @GetMapping("/daily")
    public ResponseFindDailyBudget getDailyBudget(@AuthenticationPrincipal CustomUserDetails userDetails) {
        return budgetService.getDailyBudget(userDetails.getUserPk());
    }

    // 예산 알람 수정
    @PostMapping("/alarm")
    public ResponseUpdateBudgetAlarm updateBudgetAlarm(@AuthenticationPrincipal CustomUserDetails userDetails, @RequestBody RequestUpdateBudgetAlarm requestUpdateBudgetAlarm) {
        return budgetService.updateBudgetAlarm(userDetails.getUserPk(), requestUpdateBudgetAlarm);
    }

    // 예산 가계부 조회
    @PostMapping("/detail")
    public ResponseFindHouseholdOfBudget getHouseholdOfBudget (@AuthenticationPrincipal CustomUserDetails userDetails, @RequestBody RequestFindHouseholdOfBudget requestFindHouseholdOfBudget) {
        return budgetService.getHouseholdOfBudget(userDetails.getUserPk(), requestFindHouseholdOfBudget);
    }
}
