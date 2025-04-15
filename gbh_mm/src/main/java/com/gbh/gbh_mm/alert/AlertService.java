package com.gbh.gbh_mm.alert;

import com.gbh.gbh_mm.budget.model.entity.Budget;
import com.gbh.gbh_mm.budget.model.entity.BudgetCategory;
import com.gbh.gbh_mm.budget.model.response.ResponseFindDailyBudget;
import com.gbh.gbh_mm.budget.repo.BudgetCategoryRepository;
import com.gbh.gbh_mm.budget.repo.BudgetRepository;
import com.gbh.gbh_mm.budget.service.BudgetService;
import com.gbh.gbh_mm.fcm.service.FCMService;
import com.gbh.gbh_mm.notification.service.NotificationService;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.repo.UserRepository;
import com.google.firebase.messaging.AndroidConfig;
import com.google.firebase.messaging.AndroidConfig.Priority;
import com.google.firebase.messaging.FirebaseMessaging;
import com.google.firebase.messaging.FirebaseMessagingException;
import com.google.firebase.messaging.Message;
import com.google.firebase.messaging.Notification;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Service
@RequiredArgsConstructor
public class AlertService {

    private final BudgetService budgetService;
    private final FCMService fcmService;
    private final UserRepository userRepository;
    private final BudgetCategoryRepository budgetCategoryRepository;
    private final BudgetRepository budgetRepository;
    private final NotificationService notificationService;

    public void sendNotification(String token, String title, String body)
        throws ExecutionException, InterruptedException {
        Message message = Message.builder()
            .setToken(token)
            .setNotification(Notification.builder()
                .setTitle(title)
                .setBody(body)
                .build())
            .setAndroidConfig(AndroidConfig.builder()
                .setPriority(AndroidConfig.Priority.HIGH)
                .build())
            .build();
        String response = FirebaseMessaging.getInstance().sendAsync(message).get();


//        Notification noti = Notification.builder()
//            .setTitle(title)
//            .setBody(body)
//            .build();
//
//        Message message = Message.builder()
//            .setToken(token)
//            .setNotification(noti)
//            .setAndroidConfig(AndroidConfig.builder()
//                .setPriority(Priority.HIGH)
//                .build())
//            .build();
//
//        try {
//            String response = FirebaseMessaging.getInstance().sendAsync(message).get();
//            System.out.println(response);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
    }

    // 오늘의 예산 알림
    @Transactional
    public void sendBudgetNotification(User user) throws InterruptedException, ExecutionException {
        ResponseFindDailyBudget dailyBudget = budgetService.getDailyBudget(user.getUserPk());
        String remainBudgetAmount = String.format("%,d", dailyBudget.getRemainBudgetAmount());
        String dailyBudgetAmount = String.format("%,d", dailyBudget.getDailyBudgetAmount());
        String title = "오늘의 예산";
        String body = "이번 달 남은 예산 " + remainBudgetAmount + " 원!" + "오늘은 " + dailyBudgetAmount
            + " 원까지만 써 보세요.";
        Notification noti = Notification.builder()
            .setTitle(title)
            .setBody(body)
            .build();

        Message message = Message.builder()
            .setToken(user.getFcmToken())
            .setNotification(noti)
            .build();

        System.out.println(fcmService.sendNotification(message));
        notificationService.saveToredis(String.valueOf(user.getUserPk()), title, body);
    }

    // 오늘의 지출 퍼센트 알림
    @Transactional
    public void sendExpendNotification(String userPk, String token, String category,
        int expendPercent) throws InterruptedException, ExecutionException {

        String title = "예산 초과 경고!!";
        String body = "조심하세요! 이번 달 " + category + " 예산의 " + expendPercent + " % " + "이상을 지출 했어요!";
        Notification noti = Notification.builder()
            .setTitle(title)
            .setBody(body)
            .build();

        Message message = Message.builder()
            .setToken(token)
            .setNotification(noti)
            .build();

        System.out.println(fcmService.sendNotification(message));
        notificationService.saveToredis(userPk, title, body);
    }

    // 지출 비율 알림 로직
    public void expendNotificationProcess() throws InterruptedException, ExecutionException {
        List<User> userList = userRepository.findAll();
        for (User user : userList) {
            if (user.getFcmToken() != null && !user.getFcmToken().isEmpty()) {
                Budget budget = budgetRepository.findAllByUser_UserPkOrderByBudgetPkDesc(
                    user.getUserPk()).get(0);
                if (budget != null) {
                    List<BudgetCategory> budgetCategoryList = budgetCategoryRepository.findAllByBudget_BudgetPk(
                        budget.getBudgetPk()).stream().toList();
                    for (BudgetCategory budgetCategory : budgetCategoryList) {
                        String budgetCategoryName = budgetCategory.getBudgetCategoryName();
                        float budgetCategoryExpendPercent =
                            (float) budgetCategory.getBudgetExpendAmount()
                                / (float) budgetCategory.getBudgetCategoryPrice();
                        int percent = 0;
                        if (budgetCategoryExpendPercent < 0.3f) {
                            continue;
                        } else if (budgetCategoryExpendPercent < 0.5f) {
                            percent = 30;
                        } else if (budgetCategoryExpendPercent < 0.5f) {
                            percent = 50;
                        } else {
                            percent = 70;
                        }

                        sendExpendNotification(String.valueOf(user.getUserPk()), user.getFcmToken(),
                            budgetCategoryName, percent);
                    }
                }
            }
        }
    }
}
