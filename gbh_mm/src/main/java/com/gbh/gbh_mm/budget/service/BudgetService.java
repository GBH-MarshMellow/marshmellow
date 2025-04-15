package com.gbh.gbh_mm.budget.service;

import com.gbh.gbh_mm.budget.model.entity.Budget;
import com.gbh.gbh_mm.budget.model.entity.BudgetCategory;
import com.gbh.gbh_mm.budget.model.request.RequestCreateBudget;
import com.gbh.gbh_mm.budget.model.request.RequestFindHouseholdOfBudget;
import com.gbh.gbh_mm.budget.model.request.RequestUpdateBudgetAlarm;
import com.gbh.gbh_mm.budget.model.request.RequestUpdateBudgetCategory;
import com.gbh.gbh_mm.budget.model.response.*;
import com.gbh.gbh_mm.budget.repo.BudgetCategoryRepository;
import com.gbh.gbh_mm.budget.repo.BudgetRepository;
import com.gbh.gbh_mm.common.exception.CustomException;
import com.gbh.gbh_mm.common.exception.ErrorCode;
import com.gbh.gbh_mm.household.model.entity.Household;
import com.gbh.gbh_mm.household.repo.HouseholdRepository;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.repo.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalAdjusters;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class BudgetService {

    private final UserRepository userRepository;
    private final BudgetRepository budgetRepository;
    private final BudgetCategoryRepository budgetCategoryRepository;
    private final HouseholdRepository householdRepository;

    // 전체 예산 조회
    public ResponseFindBudgetList getBudgetList(Long userPk) {
        List<Budget> budgets = budgetRepository.findAllByUser_UserPkOrderByBudgetPkDesc(userPk);

        if (budgets.isEmpty()) {
            throw new CustomException(ErrorCode.RESOURCE_NOT_FOUND);
        }

        DateTimeFormatter dashFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        DateTimeFormatter noDashFormatter = DateTimeFormatter.ofPattern("yyyyMMdd");

        List<ResponseFindBudgetList.BudgetData> budgetDataList = budgets.stream()
                .map(budget -> {
                    List<BudgetCategory> budgetCategories = budgetCategoryRepository.findAllByBudget_BudgetPk(budget.getBudgetPk());
                    List<ResponseFindBudgetList.BudgetData.BudgetCategoryData> categoryDataList = budgetCategories.stream()
                            .map(category -> {
                                String startDateFormatted = LocalDate.parse(budget.getStartDate(), dashFormatter).format(noDashFormatter);
                                String endDateFormatted = LocalDate.parse(budget.getEndDate(), dashFormatter).format(noDashFormatter);

                                long totalAmount = householdRepository.findHouseholdsByBudget(userPk, startDateFormatted, endDateFormatted, category.getBudgetCategoryName())
                                        .stream().mapToLong(Household::getHouseholdAmount).sum();

                                return ResponseFindBudgetList.BudgetData.BudgetCategoryData.builder()
                                    .budgetCategoryPk(category.getBudgetCategoryPk())
                                    .budgetCategoryName(category.getBudgetCategoryName())
                                    .budgetCategoryPrice(category.getBudgetCategoryPrice())
                                    .budgetExpendAmount(totalAmount)
//                                    .budgetExpendAmount(category.getBudgetExpendAmount())
                                    .budgetExpendPercent(
                                            (float) Math.round(
                                                    (float) totalAmount / (float) category.getBudgetCategoryPrice() * 100) / 100.0
                                    )
                                    .build();
                            })
                            .collect(Collectors.toList());

                    return ResponseFindBudgetList.BudgetData.builder()
                            .budgetPk(budget.getBudgetPk())
                            .budgetAmount(budget.getBudgetAmount())
                            .startDate(budget.getStartDate())
                            .endDate(budget.getEndDate())
                            .budgetCategoryList(categoryDataList) // 추가된 부분
                            .build();
                })
                .collect(Collectors.toList());
        return ResponseFindBudgetList.builder()
                .message("예산 리스트 조회")
                .budgetList(budgetDataList)
                .build();
    }

    // 예산 생성
    @Transactional
    public ResponseCreateBudget createBudget(Long userPk, RequestCreateBudget requestCreateBudget) {

        User user = userRepository.findById(userPk).
                orElseThrow(() -> new CustomException(ErrorCode.CHILD_NOT_FOUND));

        Budget budget = new Budget();

        int salaryDate = user.getSalaryDate();
        LocalDate today = LocalDate.now();

        LocalDate startDate;
        LocalDate endDate;

        // 1월 말일 때 로직 추가 전
        // 오늘이 월급일 전일 때 -> 이전 달부터의 예산 생성
        if (today.getDayOfMonth() < salaryDate) {
            LocalDate previousMonth = today.minusMonths(1);
            int lastDayOfPreviousMonth = previousMonth.with(TemporalAdjusters.lastDayOfMonth()).getDayOfMonth();
            int validSalaryDate = Math.min(salaryDate, lastDayOfPreviousMonth);
            startDate = previousMonth.withDayOfMonth(validSalaryDate);
            endDate = today.withDayOfMonth(salaryDate - 1);
        }
        // 오늘이 월급날 이상일 때 -> 이번 달부터 예산 생성
        else {
            startDate = today.withDayOfMonth(salaryDate);
            LocalDate nextMonth = today.plusMonths(1);
            int lastDayOfNextMonth = nextMonth.with(TemporalAdjusters.lastDayOfMonth()).getDayOfMonth();
            int validEndDate = Math.min(salaryDate - 1, lastDayOfNextMonth);

            endDate = nextMonth.withDayOfMonth(validEndDate);
        }

        // Salary가 null이면 기본값 0L 사용
        long salary = Optional.ofNullable(requestCreateBudget.getSalary()).orElse(0L);

        // 카테고리별 비율 매핑
        Map<String, Float> categoryMap = Map.of(
                "고정지출", requestCreateBudget.getFixedExpense(),
                "식비/외식", requestCreateBudget.getFoodExpense(),
                "교통/자동차", requestCreateBudget.getTransportationExpense(),
                "편의점/마트", requestCreateBudget.getMarketExpense(),
                "금융", requestCreateBudget.getFinancialExpense(),
                "여가비", requestCreateBudget.getLeisureExpense(),
                "커피/디저트", requestCreateBudget.getCoffeeExpense(),
                "쇼핑", requestCreateBudget.getShoppingExpense(),
                "비상금", requestCreateBudget.getEmergencyExpense()
        );

        // 총 예산 금액 계산
        float totalPercentage = (float) categoryMap.values().stream().mapToDouble(Float::doubleValue).sum();
        long budgetAmount = Math.round(salary * totalPercentage);

        budget.setBudgetAmount(budgetAmount);
        budget.setStartDate(String.valueOf(startDate));
        budget.setEndDate(String.valueOf(endDate));
        budget.setUser(user);
        budgetRepository.save(budget);

        // 카테고리별 BudgetCategory 생성 및 저장
        List<BudgetCategory> budgetCategories = categoryMap.entrySet().stream()
                .map(entry -> {
                    BudgetCategory budgetCategory = new BudgetCategory();
                    budgetCategory.setBudget(budget);
                    budgetCategory.setBudgetCategoryName(entry.getKey());
                    budgetCategory.setBudgetCategoryPrice((long) Math.round(salary * entry.getValue()));
                    return budgetCategory;
                })
                .collect(Collectors.toList());

        budgetCategoryRepository.saveAll(budgetCategories);

        return ResponseCreateBudget.builder()
                .message("예산 생성 완료")
                .budgetPk(budget.getBudgetPk())
                .budgetAmount(budget.getBudgetAmount())
                .startDate(budget.getStartDate())
                .endDate(budget.getEndDate())
                .build();
    }

    // 세부 예산 생성
    @Transactional
    public ResponseCreateBudgetCategory createBudgetCategory(Long budgetPk, BudgetCategory budgetCategory) {

        Budget budget = budgetRepository.findById(budgetPk)
                .orElseThrow(() -> new CustomException(ErrorCode.RESOURCE_NOT_FOUND));

        budgetCategory.setBudget(budget);
        budgetCategoryRepository.save(budgetCategory);

        return ResponseCreateBudgetCategory.builder()
                .message("세부 예산 생성 완료")
                .budgetCategoryPk(budgetCategory.getBudgetCategoryPk())
                .budgetCategoryName(budgetCategory.getBudgetCategoryName())
                .budgetCategoryPrice(budgetCategory.getBudgetCategoryPrice())
                .relatedBudgetPk(budgetPk)
                .build();
    }

    // 세부 예산 조회
    public ResponseFindBudgetCategoryList getBudgetCategoryList(Long budgetPk) {

        DateTimeFormatter dashFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        DateTimeFormatter noDashFormatter = DateTimeFormatter.ofPattern("yyyyMMdd");

        Budget budget = budgetRepository.findById(budgetPk)
                .orElseThrow(() -> new CustomException(ErrorCode.RESOURCE_NOT_FOUND));

        List<ResponseFindBudgetCategoryList.BudgetCategoryData> categoryDataList =
                budgetCategoryRepository.findAllByBudget_BudgetPk(budgetPk)
                        .stream()
                        .map(budgetCategory -> {

                            String startDateFormatted = LocalDate.parse(budget.getStartDate(), dashFormatter).format(noDashFormatter);
                            String endDateFormatted = LocalDate.parse(budget.getEndDate(), dashFormatter).format(noDashFormatter);

                            long totalAmount = householdRepository.findHouseholdsByBudget(budget.getUser().getUserPk(), startDateFormatted, endDateFormatted, budgetCategory.getBudgetCategoryName())
                                    .stream().mapToLong(Household::getHouseholdAmount).sum();
                            System.out.println(budgetCategory.getBudgetCategoryName() + ": " + totalAmount);

                            return ResponseFindBudgetCategoryList.BudgetCategoryData.builder()
                                .budgetCategoryPk(budgetCategory.getBudgetCategoryPk())
                                .budgetCategoryName(budgetCategory.getBudgetCategoryName())
                                .budgetCategoryPrice(budgetCategory.getBudgetCategoryPrice())
                                .budgetExpendAmount(totalAmount)
                                .budgetExpendPercent(
                                        (float) Math.round(
                                                (double) totalAmount / (double) budgetCategory.getBudgetCategoryPrice() * 100) / 100.0
                                )
                                .build();}
                        )
                        .collect(Collectors.toList()); // 변환 결과 저장
        if (categoryDataList.isEmpty()) {
            throw new CustomException(ErrorCode.RESOURCE_NOT_FOUND);
        }
        return ResponseFindBudgetCategoryList.builder()
                .message("세부 예산 조회")
                .budgetCategoryList(categoryDataList)
                .build();

    }

    // 세부 예산 수정
    public ResponseUpdateBudgetCategory updateBudgetCategory(Long budgetCategoryPk, RequestUpdateBudgetCategory requestUpdateBudgetCategory) {
        BudgetCategory oldBudgetCategory = budgetCategoryRepository.findById(budgetCategoryPk)
                .orElseThrow(() -> new CustomException(ErrorCode.RESOURCE_NOT_FOUND));

        // 예산
        Budget budget = oldBudgetCategory.getBudget();
        Long oldBudgetAmount = budget.getBudgetAmount();

        Long oldBudgetCategoryPrice = oldBudgetCategory.getBudgetCategoryPrice();
        Long newBudgetCategoryPrice = requestUpdateBudgetCategory.getBudgetCategoryPrice();

        budget.setBudgetAmount(budget.getBudgetAmount() + newBudgetCategoryPrice - oldBudgetCategoryPrice);
        Long newBudgetAmount = budget.getBudgetAmount();
        oldBudgetCategory.setBudgetCategoryPrice(newBudgetCategoryPrice);
        budgetCategoryRepository.save(oldBudgetCategory);
        budgetRepository.save(budget);

        return ResponseUpdateBudgetCategory.builder()
                .message("세부 예산 수정 완료")
                .budgetCategoryPk(budgetCategoryPk)
                .oldBudgetCategoryPrice(oldBudgetCategoryPrice)
                .newBudgetCategoryPrice(newBudgetCategoryPrice)
                .oldBudgetAmount(oldBudgetAmount)
                .newBudgetAmount(newBudgetAmount)
                .build();


    }

    // 오늘의 예산 조회
    public ResponseFindDailyBudget getDailyBudget(Long userPk) {
        Budget budget = budgetRepository.findAllByUser_UserPkOrderByBudgetPkDesc(userPk).get(0);
        if (budget == null) {
            throw new CustomException(ErrorCode.RESOURCE_NOT_FOUND);
        }

        DateTimeFormatter dashFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
        DateTimeFormatter noDashFormatter = DateTimeFormatter.ofPattern("yyyyMMdd");

        String startDateFormatted = LocalDate.parse(budget.getStartDate(), dashFormatter).format(noDashFormatter);
        String endDateFormatted = LocalDate.parse(budget.getEndDate(), dashFormatter).format(noDashFormatter);

        long remainDay = ChronoUnit.DAYS.between(LocalDate.now(), LocalDate.parse(budget.getEndDate(), dashFormatter)) + 1;
        System.out.println("남은 일 수 = 예산 종료일 - 현재 일: " + remainDay + " = " + LocalDate.parse(budget.getEndDate()) + " - " + LocalDate.now());
        List<BudgetCategory> budgetCategories = budgetCategoryRepository.findAllByBudget_BudgetPk(budget.getBudgetPk());

        // 각 카테고리별로 가계부에서 지출을 조회하여 합산
        long totalExpendAmount = budgetCategories.stream()
                .mapToLong(category ->
                        householdRepository.findHouseholdsByBudget(
                                userPk,
                                startDateFormatted,
                                endDateFormatted,
                                category.getBudgetCategoryName()
                        ).stream().mapToLong(Household::getHouseholdAmount).sum()
                )
                .sum();

        long budgetAmount = budget.getBudgetAmount();

        return ResponseFindDailyBudget.builder()
                .message("오늘의 예산 조회")
                .budgetPk(budget.getBudgetPk())
                .budgetAmount(budgetAmount)
                .remainBudgetAmount(budgetAmount - totalExpendAmount)
                .dailyBudgetAmount((budgetAmount - totalExpendAmount) / remainDay)
                .build();
    }

    // 예산 알람 시간 수정
    public ResponseUpdateBudgetAlarm updateBudgetAlarm(Long userPk, RequestUpdateBudgetAlarm requestUpdateBudgetAlarm) {
        User user =  userRepository.findById(userPk)
                .orElseThrow(() -> new CustomException(ErrorCode.RESOURCE_NOT_FOUND));

        user.setBudgetAlarmTime(requestUpdateBudgetAlarm.getBudgetAlarmTime());
        userRepository.save(user);

        return ResponseUpdateBudgetAlarm.builder()
                .message("예산 알람 시간 수정 완료")
                .newAlarmTime(user.getBudgetAlarmTime())
                .build();
    }

    // 예산별 가계부 조회
    public ResponseFindHouseholdOfBudget getHouseholdOfBudget(long userPk, RequestFindHouseholdOfBudget requestFindHouseholdOfBudget) {
        // 가계부 리스트 조회
        List<Household> households = householdRepository.findHouseholdsByBudget(
                userPk,
                requestFindHouseholdOfBudget.getStartDate(),
                requestFindHouseholdOfBudget.getEndDate(),
                requestFindHouseholdOfBudget.getAiCategory()
        );

        return ResponseFindHouseholdOfBudget.builder()
                .message(requestFindHouseholdOfBudget.getAiCategory() + " 카테고리 가계부 조회")
                .totalNumberOfHouseholds(households.size())
                .totalAmount(households.stream().mapToLong(Household::getHouseholdAmount).sum())
                .households(households)
                .build();
    }

}