package com.gbh.gbh_mm.presentation.service;

import com.gbh.gbh_mm.alert.AlertService;
import com.gbh.gbh_mm.fcm.service.FCMService;
import com.gbh.gbh_mm.household.model.entity.Household;
import com.gbh.gbh_mm.household.model.entity.HouseholdDetailCategory;
import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import com.gbh.gbh_mm.household.repo.HouseholdDetailCategoryRepository;
import com.gbh.gbh_mm.household.repo.HouseholdRepository;
import com.gbh.gbh_mm.presentation.request.RequestHouseholdForPre;
import com.gbh.gbh_mm.presentation.request.RequestSendAlert;
import com.gbh.gbh_mm.presentation.request.RequestUpdateFcmToken;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.repo.UserRepository;
import com.gbh.gbh_mm.user.service.UserService;
import jakarta.persistence.EntityNotFoundException;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class PresentationService {

    private final UserRepository userRepository;

    private final AlertService alertService;

    private final HouseholdRepository householdRepository;
    private final HouseholdDetailCategoryRepository householdDetailCategoryRepository;

    public String sendAlert(RequestSendAlert request) {

        try {
            User user = userRepository.findByUserPk(request.getUserPk())
                .orElseThrow(() -> new RuntimeException("User Not Found"));

            String fcmToken = user.getFcmToken();
            String title = "월급 알림";
            String message = user.getUserName() + "님! 오늘은 월급날이에요! 월 예산을 확인해볼까요 :)";
            alertService.sendNotification(fcmToken, title, message);

            return "SUCCESS";
        } catch (Exception e) {
            e.printStackTrace();

            return "FAIL";
        }
    }

    public String createHousehold(RequestHouseholdForPre request) {

        try {
            User user = userRepository.findByUserPk(request.getUserPk())
                    .orElseThrow(() -> new RuntimeException("User Not Found"));
            HouseholdDetailCategory householdDetailCategory = householdDetailCategoryRepository
                    .findById(request.getHouseholdDetailCategoryPk())
                    .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 카테고리"));


            Household household = Household.builder()
                    .tradeName(request.getTradeName())
                    .tradeTime(request.getTradeTime())
                    .tradeDate(request.getTradeDate())
                    .householdAmount(request.getHouseholdAmount())
                    .householdMemo(request.getHouseholdMemo())
                    .paymentMethod(request.getPaymentMethod())
                    .paymentCancelYn("N")
                    .exceptedBudgetYn("N")
                    .user(user)
                    .householdDetailCategory(householdDetailCategory)
                    .householdClassificationCategory(HouseholdClassificationEnum.WITHDRAWAL)
                    .build();

            householdRepository.save(household);

            return "SUCCESS";
        } catch (Exception e) {
            e.printStackTrace();

            return "FAIL";
        }
    }

    public String updateFcmToken(RequestUpdateFcmToken request, CustomUserDetails customUserDetails) {
        try {
            User user = userRepository.findByUserPk(customUserDetails.getUserPk())
                .orElseThrow(() -> new RuntimeException("User Not Found"));

            user.setFcmToken(request.getToken());

            userRepository.save(user);

            return "SUCCESS";
        } catch (Exception e) {
            return "FAIL";
        }
    }
}
