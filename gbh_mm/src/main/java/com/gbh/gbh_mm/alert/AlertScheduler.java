package com.gbh.gbh_mm.alert;

import com.gbh.gbh_mm.budget.model.entity.Budget;
import com.gbh.gbh_mm.budget.model.entity.BudgetCategory;
import com.gbh.gbh_mm.budget.repo.BudgetCategoryRepository;
import com.gbh.gbh_mm.budget.repo.BudgetRepository;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.repo.UserRepository;
import lombok.AllArgsConstructor;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.YearMonth;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Component
@AllArgsConstructor
public class AlertScheduler {
    private AlertService alertService;
    private UserRepository userRepository;
    private BudgetRepository budgetRepository;
    private BudgetCategoryRepository budgetCategoryRepository;

    /* 매일 10시 */
    @Scheduled(cron = "0 0 10 * * ?")
    public void sendSalaryNotification() throws ExecutionException, InterruptedException {
        LocalDate todayDate = LocalDate.now();
        int today = todayDate.getDayOfMonth();
        YearMonth currentMonth = YearMonth.from(todayDate);
        int lastDayOfMonth = currentMonth.lengthOfMonth();

        List<User> userList = userRepository.findBySalaryDate(today);

        for (User user : userList) {
            int userSalaryDate = user.getSalaryDate();

            if (today == userSalaryDate) {
                sendNotification(user);
            } else if (userSalaryDate > lastDayOfMonth && today == lastDayOfMonth) {
                sendNotification(user);
            }
        }
    }

    private void sendNotification(User user) throws ExecutionException, InterruptedException {
        if (user.getFcmToken() != null && !user.getFcmToken().isEmpty()) {
            String title = "월급 알림";
            String body = "오늘은 월급날이에요! 예산을 짜드릴게요 :)";
            alertService.sendNotification(user.getFcmToken(), title, body);
        }
    }

    // 오늘의 예산 알림
    /* 매일 9시 */
    @Scheduled(cron = "0 0 9 * * ?", zone = "Asia/Seoul")
    public void sendBudgetNotification() throws InterruptedException, ExecutionException {

        List<User> userList = userRepository.findAll();

        for (User user : userList) {
            if (user.getFcmToken() != null && !user.getFcmToken().isEmpty()) {
                alertService.sendBudgetNotification(user);
            }
        }
    }

    // 지출 퍼센트 알림
    /* 매일 9시 6시 */
    @Scheduled(cron = "0 0 9 * * *", zone = "Asia/Seoul")
    public void sendExpendNotificationAtNine() throws InterruptedException, ExecutionException {
        expendNotificationProcess();
    }

    @Scheduled(cron = "0 0 18 * * *", zone = "Asia/Seoul")
    public void sendExpendNotificationAtEighteen() throws InterruptedException, ExecutionException {
        expendNotificationProcess();
    }

    private void expendNotificationProcess() throws InterruptedException, ExecutionException {
        List<User> userList = userRepository.findAll();
        for (User user : userList) {
            if (user.getFcmToken() != null && !user.getFcmToken().isEmpty()) {
                Budget budget = budgetRepository.findById(user.getUserPk()).orElseThrow();
                if (budget != null) {
                    List<BudgetCategory> budgetCategoryList = budgetCategoryRepository.findById(budget.getBudgetPk()).stream().toList();
                    for (BudgetCategory budgetCategory : budgetCategoryList) {
                        String budgetCategoryName = budgetCategory.getBudgetCategoryName();
                        float budgetCategoryExpendPercent = (float) budgetCategory.getBudgetExpendAmount() / (float) budgetCategory.getBudgetCategoryPrice();
                        int percent = 0;
                        if (budgetCategoryExpendPercent < 0.3f) continue;
                        else if (budgetCategoryExpendPercent < 0.5f) percent = 30;
                        else if (budgetCategoryExpendPercent < 0.5f) percent = 50;
                        else percent = 70;

                        alertService.sendExpendNotification(String.valueOf(user.getUserPk()), user.getFcmToken(), budgetCategoryName, percent);
                    }
                }
            }
        }
    }

}
