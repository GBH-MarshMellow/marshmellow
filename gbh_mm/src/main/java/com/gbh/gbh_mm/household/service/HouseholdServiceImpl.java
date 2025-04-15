package com.gbh.gbh_mm.household.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.CardAPI;
import com.gbh.gbh_mm.api.DemandDepositAPI;
import com.gbh.gbh_mm.budget.model.entity.Budget;
import com.gbh.gbh_mm.budget.model.entity.BudgetCategory;
import com.gbh.gbh_mm.budget.repo.BudgetCategoryRepository;
import com.gbh.gbh_mm.budget.repo.BudgetRepository;
import com.gbh.gbh_mm.common.exception.CustomException;
import com.gbh.gbh_mm.common.exception.ErrorCode;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindCardTransactionList;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindTransactionList;
import com.gbh.gbh_mm.household.model.dto.CardDto;
import com.gbh.gbh_mm.household.model.dto.DateGroupDto;
import com.gbh.gbh_mm.household.model.dto.DemandDepositDto;
import com.gbh.gbh_mm.household.model.dto.HouseHoldDto;
import com.gbh.gbh_mm.household.model.dto.HouseholdDetailDto;
import com.gbh.gbh_mm.household.model.dto.PaymentMethodDto;
import com.gbh.gbh_mm.household.model.entity.AiCategory;
import com.gbh.gbh_mm.household.model.entity.Household;
import com.gbh.gbh_mm.household.model.entity.HouseholdCategory;
import com.gbh.gbh_mm.household.model.entity.HouseholdDetailCategory;
import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import com.gbh.gbh_mm.household.model.vo.request.*;
import com.gbh.gbh_mm.household.model.vo.response.*;
import com.gbh.gbh_mm.household.repo.HouseholdCategoryRepository;
import com.gbh.gbh_mm.household.repo.HouseholdDetailCategoryRepository;
import com.gbh.gbh_mm.household.repo.AiCategoryRepository;
import com.gbh.gbh_mm.household.repo.HouseholdRepository;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.repo.UserRepository;
import jakarta.persistence.EntityNotFoundException;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;
import lombok.AllArgsConstructor;
import org.modelmapper.ModelMapper;
import org.modelmapper.convention.MatchingStrategies;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class HouseholdServiceImpl implements HouseholdService {

    private final HouseholdRepository householdRepository;
    private final HouseholdCategoryRepository householdCategoryRepository;
    private final HouseholdDetailCategoryRepository householdDetailCategoryRepository;
    private final AiCategoryRepository aiCategoryRepository;
    private final UserRepository userRepository;

    // 예산
    private final BudgetRepository budgetRepository;
    private final BudgetCategoryRepository budgetCategoryRepository;

    private final CardAPI cardAPI;
    private final DemandDepositAPI demandDepositAPI;

    private final ModelMapper mapper;

    @Override
    public ResponseFindHouseholdList findHouseholdList(RequestFindHouseholdList request,
        CustomUserDetails customUserDetails) {
        List<Household> householdList = householdRepository
            .findAllByTradeDateBetweenAndUser_UserPkOrderByTradeDateAsc
                (request.getStartDate(), request.getEndDate(), customUserDetails.getUserPk());

        long totalIncome = householdList.stream()
            .filter(h -> h.getHouseholdClassificationCategory()
                .equals(HouseholdClassificationEnum.DEPOSIT))
            .mapToLong(Household::getHouseholdAmount)
            .sum();

        long totalExpenditure = householdList.stream()
            .filter(h -> h.getHouseholdClassificationCategory()
                .equals(HouseholdClassificationEnum.WITHDRAWAL))
            .mapToLong(Household::getHouseholdAmount)
            .sum();

        Map<String, List<HouseholdDetailDto>> grouped = householdList.stream()
            .collect(Collectors.groupingBy(Household::getTradeDate,
                Collectors.mapping(h -> {
                    HouseholdDetailDto householdDetailDto = HouseholdDetailDto.builder()
                        .householdPk(h.getHouseholdPk())
                        .tradeName(h.getTradeName())
                        .tradeDate(h.getTradeDate())
                        .tradeTime(h.getTradeTime())
                        .householdAmount(h.getHouseholdAmount())
                        .paymentMethod(h.getPaymentMethod())
                        .paymentCancelYn(h.getPaymentCancelYn())
                        .householdCategory(h.getHouseholdDetailCategory()
                            .getHouseholdCategory()
                            .getHouseholdCategoryName())
                        .householdClassificationCategory(h.getHouseholdClassificationCategory())
                        .build();

                    return householdDetailDto;
                }, Collectors.toList())
            ));

        List<DateGroupDto> households = grouped.entrySet().stream()
            .sorted((a, b) -> b.getKey().compareTo(a.getKey()))
            .map(entry -> {
                DateGroupDto dto = DateGroupDto.builder()
                    .date(entry.getKey())
                    .list(entry.getValue())
                    .build();

                return dto;
            })
            .collect(Collectors.toList());

        ResponseFindHouseholdList response = ResponseFindHouseholdList.builder()
            .totalIncome(totalIncome)
            .totalExpenditure(totalExpenditure)
            .householdList(households)
            .build();

        return response;
    }

    @Override
    public ResponseCreateHousehold createHousehold(RequestCreateHousehold request,
        CustomUserDetails customUserDetails) {
        AiCategory aiCategory = aiCategoryRepository.findById(9)
            .orElseThrow(() -> new EntityNotFoundException("미분류 찾을 수 없음"));
        HouseholdCategory householdCategory = householdCategoryRepository
            .findById(19)
            .orElseThrow(() -> new EntityNotFoundException("미분류 찾을 수 없음"));

        mapper.getConfiguration().setMatchingStrategy(MatchingStrategies.STRICT);
        Household household = mapper.map(request, Household.class);
        User user = userRepository.findById(customUserDetails.getUserPk())
            .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 회원"));
        HouseholdDetailCategory householdDetailCategory =
            householdDetailCategoryRepository
                .findById(request.getHouseholdDetailCategoryPk())
                .orElse(HouseholdDetailCategory.builder()
                    .householdDetailCategoryPk(118)
                    .aiCategory(aiCategory)
                    .householdCategory(householdCategory)
                    .build());

        if (householdDetailCategory == null) {
            householdDetailCategory = householdDetailCategoryRepository
                .findByHouseholdDetailCategory("미분류");
        }

        household.setUser(user);
        household.setHouseholdClassificationCategory(request.getHouseholdClassification());
        household.setHouseholdDetailCategory(householdDetailCategory);
        household.setPaymentCancelYn("N");

        Household savedHousehold = householdRepository.save(household);

        ResponseCreateHousehold response = ResponseCreateHousehold.builder()
            .householdPk(savedHousehold.getHouseholdPk())
            .tradeName(savedHousehold.getTradeName())
            .tradeDate(savedHousehold.getTradeDate())
            .tradeTime(savedHousehold.getTradeTime())
            .householdAmount(savedHousehold.getHouseholdAmount())
            .householdMemo(savedHousehold.getHouseholdMemo())
            .paymentMethod(savedHousehold.getPaymentMethod())
            .paymentCancelYn(savedHousehold.getPaymentCancelYn())
            .exceptedBudgetYn(savedHousehold.getExceptedBudgetYn())
            .householdCategory(savedHousehold.getHouseholdDetailCategory()
                .getHouseholdCategory().getHouseholdCategoryName())
            .householdDetailCategory
                (savedHousehold.getHouseholdDetailCategory().getHouseholdDetailCategory())
            .householdClassificationCategory
                (savedHousehold.getHouseholdClassificationCategory())
            .build();

        return response;
    }

    @Override
    public ResponseFindHousehold findHousehold(RequestFindHousehold request) {
        Household household = householdRepository.findById(request.getHouseholdPk())
            .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 가계부입니다."));

        ResponseFindHousehold response = ResponseFindHousehold.builder()
            .householdPk(household.getHouseholdPk())
            .tradeName(household.getTradeName())
            .tradeDate(household.getTradeDate())
            .tradeTime(household.getTradeTime())
            .householdAmount(household.getHouseholdAmount())
            .householdMemo(household.getHouseholdMemo())
            .paymentMethod(household.getPaymentMethod())
            .paymentCancelYn(household.getPaymentCancelYn())
            .exceptedBudgetYn(household.getExceptedBudgetYn())
            .householdCategory(household.getHouseholdDetailCategory()
                .getHouseholdCategory().getHouseholdCategoryName())
            .householdDetailCategoryPk
                (household.getHouseholdDetailCategory().getHouseholdDetailCategoryPk())
            .householdDetailCategory
                (household.getHouseholdDetailCategory().getHouseholdDetailCategory())
            .householdClassificationCategory
                (household.getHouseholdClassificationCategory())
            .build();

        return response;
    }

    @Override
    public ResponseUpdateHousehold updateHousehold(RequestUpdateHousehold request) {
        Household household = householdRepository.findById(request.getHouseholdPk())
            .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 가계부"));

        if (request.getHouseholdAmount() != null) {
            household.setHouseholdAmount(request.getHouseholdAmount());
        }

        if (request.getExceptedBudgetYn() != null) {
            household.setExceptedBudgetYn(request.getExceptedBudgetYn());
        }

        if (request.getHouseholdMemo() != null) {
            household.setHouseholdMemo(request.getHouseholdMemo());
        }

        HouseholdDetailCategory householdDetailCategory = householdDetailCategoryRepository
            .findById(request.getHouseholdDetailCategoryPk())
            .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 분류"));
        household.setHouseholdDetailCategory(householdDetailCategory);


        householdRepository.save(household);

        mapper.getConfiguration().setMatchingStrategy(MatchingStrategies.STRICT);
        ResponseUpdateHousehold response = mapper.map(household, ResponseUpdateHousehold.class);

        response.setHouseholdCategory(household.getHouseholdDetailCategory()
            .getHouseholdCategory().getHouseholdCategoryName());
        response.setHouseholdDetailCategory
            (household.getHouseholdDetailCategory().getHouseholdDetailCategory());
        response.setHouseholdClassificationCategory
            (household.getHouseholdClassificationCategory());

        return response;
    }

    @Override
    public ResponseDeleteHousehold deleteHousehold(RequestDeleteHousehold request) {
        ResponseDeleteHousehold response = new ResponseDeleteHousehold();
        try {
            Optional<Household> household = householdRepository.findById(request.getHouseholdPk());
            householdRepository.delete(household.get());
            response.setMessage("삭제 성공");
        } catch (Exception e) {
            response.setMessage("삭제 실패");
        }

        return response;
    }

    @Override
    public ResponseFindTransactionDataList findTransactionDataList(
        CustomUserDetails customUserDetails) {
        mapper.getConfiguration().setMatchingStrategy(MatchingStrategies.STRICT);

        List<Household> householdList = householdRepository
            .findTop2ByUser_UserPkOrderByTradeDateDesc(customUserDetails.getUserPk());

        LocalDate currentDate = LocalDate.now();

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd");
        String currentString = currentDate.format(formatter);
        String lastDate = "";
        if (householdList.size() > 0) {
            /* 마지막일 ~ 현재일 */
            lastDate = householdList.getFirst().getTradeDate();
        } else {
            /* 전부 조회 */
            lastDate = "20200101";
        }

        List<Household> houseHolds = new ArrayList<>();

        try {
            Optional<User> user = userRepository.findById(customUserDetails.getUserPk());

            Map<String, Object> cardApiData =
                cardAPI.findUserCardList(customUserDetails.getUserKey());
            Map<String, Object> cardResponseData =
                (Map<String, Object>) cardApiData.get("apiResponse");
            List<Map<String, Object>> cardRecData =
                (List<Map<String, Object>>) cardResponseData.get("REC");

            List<CardDto> cardDtoList = new ArrayList<>();
            for (Map<String, Object> cardRecDatum : cardRecData) {
                CardDto cardDto = mapper.map(cardRecDatum, CardDto.class);
                cardDtoList.add(cardDto);
            }

            Map<String, Object> demandDepositApiData =
                demandDepositAPI.findDemandDepositAccountList(customUserDetails.getUserKey());
            Map<String, Object> demandDepositReponseData =
                (Map<String, Object>) demandDepositApiData.get("apiResponse");
            List<Map<String, Object>> demandDepositRecData =
                (List<Map<String, Object>>) demandDepositReponseData.get("REC");

            /* 조회한 카드 목록에서 하나씩 분리 */
            for (CardDto card : cardDtoList) {
                RequestFindCardTransactionList requestCardTransaction =
                    new RequestFindCardTransactionList();

                requestCardTransaction.setCardNo(card.getCardNo());
                requestCardTransaction.setCvc(card.getCvc());
                requestCardTransaction.setEndDate(currentString);
                requestCardTransaction.setStartDate(lastDate);
                requestCardTransaction.setUserKey(customUserDetails.getUserKey());

                /* 해당 카드 거래내역 조회 */
                Map<String, Object> cardTransactionApiData =
                    cardAPI.findTransactionList(requestCardTransaction);

                Map<String, Object> cardTransactionReponseData =
                    (Map<String, Object>) cardTransactionApiData.get("apiResponse");
                Map<String, Object> cardTransactionRecData =
                    (Map<String, Object>) cardTransactionReponseData.get("REC");
                List<Map<String, Object>> cardTransactionList =
                    (List<Map<String, Object>>) cardTransactionRecData.get("transactionList");


                /* 해당 카드 거래내역 매핑 */
                if (cardTransactionList != null) {
                    for (Map<String, Object> cardTransactionRecDatum : cardTransactionList) {
                        /* 해당 카드 거래내역 Response에 추가 */
                        Household household = Household.builder()
                            .tradeName((String) cardTransactionRecDatum.get("merchantName"))
                            .tradeDate((String) cardTransactionRecDatum.get("transactionDate"))
                            .tradeTime((String) cardTransactionRecDatum.get("transactionTime"))
                            .householdAmount(
                                Integer.parseInt(
                                    (String) cardTransactionRecDatum.get("transactionBalance")))
                            .paymentMethod(card.getCardName())
                            .exceptedBudgetYn("N")
                            .householdClassificationCategory(HouseholdClassificationEnum.WITHDRAWAL)
                            .user(user.get())
                            .build();
                        if (cardTransactionRecDatum.get("cardStatus").equals("승인")) {
                            household.setPaymentCancelYn("N");
                        } else {
                            household.setPaymentCancelYn("Y");
                        }
                        houseHolds.add(household);
                    }
                }
            }

            List<DemandDepositDto> demandDepositDtoList = new ArrayList<>();
            for (Map<String, Object> demandDepositRecDatum : demandDepositRecData) {
                DemandDepositDto DemandDepositDto =
                    mapper.map(demandDepositRecDatum, DemandDepositDto.class);
                demandDepositDtoList.add(DemandDepositDto);
            }

            for (DemandDepositDto demandDepositDto : demandDepositDtoList) {
                RequestFindTransactionList requestFindTransactionList =
                    RequestFindTransactionList.builder()
                        .accountNo(demandDepositDto.getAccountNo())
                        .startDate(lastDate)
                        .endDate(currentString)
                        .transactionType("A")
                        .orderByType("ASC")
                        .userKey(customUserDetails.getUserKey())
                        .build();

                Map<String, Object> demandDepositTransactionData =
                    demandDepositAPI.findTransactionList(requestFindTransactionList);
                Map<String, Object> demandDepositTransactionApiResponse =
                    (Map<String, Object>) demandDepositTransactionData.get("apiResponse");
                Map<String, Object> demandDepositTransactionRecData =
                    (Map<String, Object>) demandDepositTransactionApiResponse.get("REC");
                List<Map<String, Object>> demandDepositTransactionList =
                    (List<Map<String, Object>>) demandDepositTransactionRecData.get("list");

                if (demandDepositTransactionList != null) {

                    for (Map<String, Object> demandDepositTransaction : demandDepositTransactionList) {
                        Household household = Household.builder()
                            .tradeName((String) demandDepositTransaction.get("transactionSummary"))
                            .tradeDate((String) demandDepositTransaction.get("transactionDate"))
                            .tradeTime((String) demandDepositTransaction.get("transactionTime"))
                            .householdAmount(
                                Integer.parseInt(
                                    (String) demandDepositTransaction.get("transactionBalance")))
                            .paymentMethod(demandDepositDto.getAccountName())
                            .exceptedBudgetYn("N")
                            .paymentCancelYn("N")
                            .user(user.get())
                            .build();

                        if (demandDepositTransaction.get("transactionType").equals("1")) {
                            household.setHouseholdClassificationCategory(
                                HouseholdClassificationEnum.DEPOSIT);
                        } else {
                            household.setHouseholdClassificationCategory(
                                HouseholdClassificationEnum.TRANSFER);
                        }
                        houseHolds.add(household);
                    }
                }
            }
            ResponseFindTransactionDataList responseHousehold = new ResponseFindTransactionDataList();
            responseHousehold.setHouseholdList(houseHolds);

            return responseHousehold;
        } catch (JsonProcessingException e) {

            throw new RuntimeException(e);
        }
    }

    @Override
    public ResponseCreateHouseholdList createHouseholdList(RequestCreateHouseholdList request) {

        List<HouseHoldDto> householdDtoList = request.getTransactionList();
        List<Household> householdList = new ArrayList<>();
        for (HouseHoldDto householdDto : householdDtoList) {
            String category = householdDto.getCategory().replaceAll("\\s+", "");

            HouseholdDetailCategory householdDetailCategory =
                householdDetailCategoryRepository.findByHouseholdDetailCategory(category);
            String tradeTimeSec = householdDto.getTradeTime();
            String tradeTime = tradeTimeSec.substring(0, tradeTimeSec.length() - 2);

            if (householdDetailCategory == null) {
                householdDetailCategory = householdDetailCategoryRepository
                    .findByHouseholdDetailCategory("미분류");
            }

            Household household = Household.builder()
                .tradeName(householdDto.getTradeName())
                .tradeDate(householdDto.getTradeDate())
                .tradeTime(tradeTime)
                .householdAmount(householdDto.getHouseholdAmount())
                .paymentMethod(householdDto.getPaymentMethod())
                .paymentCancelYn(householdDto.getPaymentCancelYn())
                .exceptedBudgetYn(householdDto.getExceptedBudgetYn())
                .user(householdDto.getUser())
                .householdDetailCategory(householdDetailCategory)
                .householdClassificationCategory(householdDto.getHouseholdClassificationCategory())
                .build();

            try {
                householdRepository.save(household);
                householdList.add(household);
            } catch (Exception e) {
                e.printStackTrace();
            }

            // 입출금 내역 예산 반영
            if (household.getHouseholdClassificationCategory().equals("WITHDRAWAL")) {

                Budget budget =
                        budgetRepository.
                                findAllByUser_UserPkOrderByBudgetPkDesc(household.getUser().getUserPk()).get(0);

                DateTimeFormatter budgetFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
                DateTimeFormatter householdFormatter = DateTimeFormatter.ofPattern("yyyyMMdd");

                LocalDate budgetStartDate = LocalDate.parse(budget.getStartDate(), budgetFormatter);
                LocalDate budgetEndDate = LocalDate.parse(budget.getEndDate(), budgetFormatter);
                LocalDate tradeDate = LocalDate.parse(household.getTradeDate(), householdFormatter);

                // if 예산 시작일 < 거래일 < 예산 종료일
                if (!tradeDate.isBefore(budgetStartDate) && !tradeDate.isAfter(budgetEndDate)) {
                    String aiCategoryName = householdDetailCategory.getAiCategory().getAiCategory();
                    List<BudgetCategory> budgetCategories =
                            budgetCategoryRepository.findAllByBudget_BudgetPk(budget.getBudgetPk());


                    for (BudgetCategory budgetCategory : budgetCategories) {
                        if (budgetCategory.getBudgetCategoryName().equals(aiCategoryName)) {
                            budgetCategory.setBudgetExpendAmount((
                                    budgetCategory.getBudgetExpendAmount() + householdDto.getHouseholdAmount()
                            ));
                            budgetCategoryRepository.save(budgetCategory);
                            break;
                        }
                    }
                }

            }

        }



        LocalDate currentDate = LocalDate.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd");
        String currentString = currentDate.format(formatter);
        String lastDate = "20200101";

        if (!householdList.isEmpty()) {
            long userPk = householdList.get(0).getUser().getUserPk();
            List<Household> afterHouseholdList = householdRepository
                .findAllByTradeDateBetweenAndUser_UserPkOrderByTradeDateAsc(lastDate, currentString,
                    userPk);

            long totalIncome = afterHouseholdList.stream()
                .filter(h -> h.getHouseholdClassificationCategory()
                    .equals(HouseholdClassificationEnum.DEPOSIT))
                .mapToLong(Household::getHouseholdAmount)
                .sum();

            long totalExpenditure = afterHouseholdList.stream()
                .filter(h -> h.getHouseholdClassificationCategory()
                    .equals(HouseholdClassificationEnum.WITHDRAWAL))
                .mapToLong(Household::getHouseholdAmount)
                .sum();

            Map<String, List<HouseholdDetailDto>> grouped = afterHouseholdList.stream()
                .collect(Collectors.groupingBy(Household::getTradeDate,
                    Collectors.mapping(h -> {
                        HouseholdDetailDto householdDetailDto = HouseholdDetailDto.builder()
                            .householdPk(h.getHouseholdPk())
                            .tradeName(h.getTradeName())
                            .tradeDate(h.getTradeDate())
                            .tradeTime(h.getTradeTime())
                            .householdAmount(h.getHouseholdAmount())
                            .paymentMethod(h.getPaymentMethod())
                            .paymentCancelYn(h.getPaymentCancelYn())
                            .householdCategory(h.getHouseholdDetailCategory()
                                .getHouseholdCategory()
                                .getHouseholdCategoryName())
                            .householdClassificationCategory(h.getHouseholdClassificationCategory())
                            .build();

                        return householdDetailDto;
                    }, Collectors.toList())
                ));

            List<DateGroupDto> households = grouped.entrySet().stream()
                .sorted((a, b) -> b.getKey().compareTo(a.getKey()))
                .map(entry -> {
                    DateGroupDto dto = DateGroupDto.builder()
                        .date(entry.getKey())
                        .list(entry.getValue())
                        .build();

                    return dto;
                })
                .collect(Collectors.toList());

            ResponseCreateHouseholdList response = ResponseCreateHouseholdList.builder()
                .totalIncome(totalIncome)
                .totalExpenditure(totalExpenditure)
                .houseHoldList(households)
                .build();

            return response;
        }

        return null;
    }

    @Override
    public ResponseSearchHousehold searchHousehold(RequestSearchHousehold request,
        CustomUserDetails customUserDetails) {
        List<Household> householdList = householdRepository.searchHousehold
            (request.getStartDate(), request.getEndDate(), customUserDetails.getUserPk(),
                request.getKeyword());

        Map<String, List<HouseholdDetailDto>> grouped = householdList.stream()
            .collect(Collectors.groupingBy(Household::getTradeDate,
                Collectors.mapping(h -> {
                    HouseholdDetailDto householdDetailDto = HouseholdDetailDto.builder()
                        .householdPk(h.getHouseholdPk())
                        .tradeName(h.getTradeName())
                        .tradeDate(h.getTradeDate())
                        .tradeTime(h.getTradeTime())
                        .householdAmount(h.getHouseholdAmount())
                        .paymentMethod(h.getPaymentMethod())
                        .paymentCancelYn(h.getPaymentCancelYn())
                        .householdCategory(h.getHouseholdDetailCategory()
                            .getHouseholdCategory()
                            .getHouseholdCategoryName())
                        .householdClassificationCategory(h.getHouseholdClassificationCategory())
                        .build();

                    return householdDetailDto;
                }, Collectors.toList())
            ));

        List<DateGroupDto> households = grouped.entrySet().stream()
            .sorted((a, b) -> b.getKey().compareTo(a.getKey()))
            .map(entry -> {
                DateGroupDto dto = DateGroupDto.builder()
                    .date(entry.getKey())
                    .list(entry.getValue())
                    .build();

                return dto;
            })
            .collect(Collectors.toList());

        ResponseSearchHousehold response = ResponseSearchHousehold.builder()
            .householdList(households)
            .build();

        return response;
    }

    @Override
    public ResponseFilterHousehold filterHousehold(RequestFilterHousehold request,
        CustomUserDetails customUserDetails) {
        List<Household> householdList = householdRepository
            .findAllByTradeDateBetweenAndUser_UserPkAndHouseholdClassificationCategory
                (request.getStartDate(), request.getEndDate(), customUserDetails.getUserPk(),
                    request.getClassification());

        long total = householdList.stream()
            .filter(h -> h.getHouseholdClassificationCategory()
                .equals(request.getClassification()))
            .mapToLong(Household::getHouseholdAmount)
            .sum();

        Map<String, List<HouseholdDetailDto>> grouped = householdList.stream()
            .collect(Collectors.groupingBy(Household::getTradeDate,
                Collectors.mapping(h -> {
                    HouseholdDetailDto householdDetailDto = HouseholdDetailDto.builder()
                        .householdPk(h.getHouseholdPk())
                        .tradeName(h.getTradeName())
                        .tradeDate(h.getTradeDate())
                        .tradeTime(h.getTradeTime())
                        .householdAmount(h.getHouseholdAmount())
                        .paymentMethod(h.getPaymentMethod())
                        .paymentCancelYn(h.getPaymentCancelYn())
                        .householdCategory(h.getHouseholdDetailCategory()
                            .getHouseholdCategory()
                            .getHouseholdCategoryName())
                        .householdClassificationCategory(h.getHouseholdClassificationCategory())
                        .build();

                    return householdDetailDto;
                }, Collectors.toList())
            ));

        List<DateGroupDto> households = grouped.entrySet().stream()
            .sorted((a, b) -> b.getKey().compareTo(a.getKey()))
            .map(entry -> {
                DateGroupDto dto = DateGroupDto.builder()
                    .date(entry.getKey())
                    .list(entry.getValue())
                    .build();

                return dto;
            })
            .collect(Collectors.toList());

        ResponseFilterHousehold response = ResponseFilterHousehold.builder()
            .total(total)
            .householdList(households)
            .build();

        return response;
    }

    @Override
    public ResponsePaymentMethodList findPaymentMethodList(CustomUserDetails customUserDetails) {

        try {
            Map<String, Object> cardResponseData = cardAPI
                .findUserCardList(customUserDetails.getUserKey());
            Map<String, Object> accountResponseData = demandDepositAPI
                .findDemandDepositAccountList(customUserDetails.getUserKey());

            Map<String, Object> cardApiData =
                (Map<String, Object>) cardResponseData.get("apiResponse");
            Map<String, Object> accountApiData =
                (Map<String, Object>) accountResponseData.get("apiResponse");
            List<Map<String, Object>> cardRecData =
                (List<Map<String, Object>>) cardApiData.get("REC");
            List<Map<String, Object>> accountRecData =
                (List<Map<String, Object>>) accountApiData.get("REC");

            List<PaymentMethodDto> paymentMethodDtoList = new ArrayList<>();
            for (Map<String, Object> cardRecDatum : cardRecData) {
                PaymentMethodDto paymentMethodDto = PaymentMethodDto.builder()
                    .bankCode((String) cardRecDatum.get("cardIssuerCode"))
                    .bankName((String) cardRecDatum.get("cardIssuerName"))
                    .paymentType("CARD")
                    .paymentMethod((String) cardRecDatum.get("cardName"))
                    .build();

                paymentMethodDtoList.add(paymentMethodDto);
            }

            for (Map<String, Object> accountRecDatum : accountRecData) {
                PaymentMethodDto paymentMethodDto = PaymentMethodDto.builder()
                    .bankCode((String) accountRecDatum.get("bankCode"))
                    .bankName((String) accountRecDatum.get("bankName"))
                    .paymentType("ACCOUNT")
                    .paymentMethod((String) accountRecDatum.get("accountName"))
                    .build();

                paymentMethodDtoList.add(paymentMethodDto);
            }

            ResponsePaymentMethodList response = new ResponsePaymentMethodList();

            response.setPaymentMethodList(paymentMethodDtoList);

            return response;

        } catch (JsonProcessingException e) {
            return null;
        }
    }

    @Override
    public ResponseAiAvg findAiAvg(CustomUserDetails customUserDetails) {
        User user = userRepository.findByUserPk(customUserDetails.getUserPk())
                .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

//        List<Household> householdList = householdRepository
//                .findAllByUser_UserPkAndHouseholdClassificationCategoryOrderByTradeDateAsc
//                        (customUserDetails.getUserPk(), HouseholdClassificationEnum.WITHDRAWAL);
        List<Household> householdList = householdRepository
                .findAllWithDetailAndAiCategoryAndHouseholdCategory(customUserDetails.getUserPk(), HouseholdClassificationEnum.WITHDRAWAL);

        long fixedAvg = 0;
        long foodAvg = 0;
        long trafficAvg = 0;
        long martAvg = 0;
        long bankAvg = 0;
        long leisureAvg = 0;
        long coffeeAvg = 0;
        long shoppingAvg = 0;
        long emergencyAvg = 0;

        for (Household household : householdList) {
            switch (household.getHouseholdDetailCategory().getAiCategory().getAiCategoryPk()) {
                case 1:
                    fixedAvg += household.getHouseholdAmount();
                    break;
                case 2:
                    foodAvg += household.getHouseholdAmount();
                    break;
                case 3:
                    trafficAvg += household.getHouseholdAmount();
                    break;
                case 4:
                    martAvg += household.getHouseholdAmount();
                    break;
                case 5:
                    bankAvg += household.getHouseholdAmount();
                    break;
                case 6:
                    leisureAvg += household.getHouseholdAmount();
                    break;
                case 7:
                    coffeeAvg += household.getHouseholdAmount();
                    break;
                case 8:
                    shoppingAvg += household.getHouseholdAmount();
                    break;
                case 9:
                    emergencyAvg += household.getHouseholdAmount();
                    break;
            }
        }

        String startDateStr = householdList.get(0).getTradeDate();
        String endDateStr = householdList.get(householdList.size() - 1).getTradeDate();

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd");
        LocalDate startDate = LocalDate.parse(startDateStr, formatter);
        LocalDate endDate = LocalDate.parse(endDateStr, formatter);

        long totalDays = ChronoUnit.DAYS.between(startDate, endDate);

        double monthDiff = totalDays / 30.44;

        double totalSalary = user.getSalaryAmount() * monthDiff;

        System.out.println(totalSalary);
        System.out.println(totalSalary);

        System.out.println(monthDiff);
        System.out.println(monthDiff);

        System.out.println(trafficAvg);
        System.out.println(trafficAvg);


        ResponseAiAvg response = ResponseAiAvg.builder()
                .salary(user.getSalaryAmount())
                .fixedAvg(fixedAvg/totalSalary)
                .foodAvg(foodAvg/totalSalary)
                .trafficAvg(trafficAvg/totalSalary)
                .martAvg(martAvg/totalSalary)
                .bankAvg(bankAvg/totalSalary)
                .leisureAvg(leisureAvg/totalSalary)
                .coffeeAvg(coffeeAvg/totalSalary)
                .shoppingAvg(shoppingAvg/totalSalary)
                .emergencyAvg(emergencyAvg/totalSalary)
                .build();

        return response;
    }

    @Override
    public Map<String, Long> findMonthlyWithdrawalMap(Long userPk, int salaryDate) {
        Map<String, Long> monthlySpendingMap = new LinkedHashMap<>();
        DateTimeFormatter yyyyMM = DateTimeFormatter.ofPattern("yyyyMM");
        DateTimeFormatter yyyyMMdd = DateTimeFormatter.ofPattern("yyyyMMdd");

        LocalDate now = LocalDate.now();

        for (int i = 11; i >= 0; i--) {
            LocalDate base = now.minusMonths(i).withDayOfMonth(1);
            int safeSalaryDate = Math.min(salaryDate, base.lengthOfMonth());

            LocalDate start = base.withDayOfMonth(safeSalaryDate);
            LocalDate end = start.plusMonths(1).minusDays(1);

            String startStr = start.format(yyyyMMdd);
            String endStr = end.format(yyyyMMdd);
            String key = start.format(yyyyMM);

            List<Household> list = householdRepository
                    .findAllByTradeDateBetweenAndUser_UserPkOrderByTradeDateAsc(startStr, endStr, userPk);

            long total = list.stream()
                    .filter(h -> h.getHouseholdClassificationCategory() == HouseholdClassificationEnum.WITHDRAWAL)
                    .mapToLong(Household::getHouseholdAmount)
                    .sum();

            monthlySpendingMap.put(key, total);
        }

        return monthlySpendingMap;
    }
}
