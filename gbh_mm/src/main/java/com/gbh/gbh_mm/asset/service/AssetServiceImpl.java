package com.gbh.gbh_mm.asset.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.alert.AlertService;
import com.gbh.gbh_mm.api.*;
import com.gbh.gbh_mm.asset.RequestDecodeTest;
import com.gbh.gbh_mm.asset.ResponseAuthTest;
import com.gbh.gbh_mm.asset.model.dto.CardListDto;
import com.gbh.gbh_mm.asset.model.dto.CardTransactionDto;
import com.gbh.gbh_mm.asset.model.dto.DemandDepositListDto;
import com.gbh.gbh_mm.asset.model.dto.DepositListDto;
import com.gbh.gbh_mm.asset.model.dto.DemandDepositTransactionDto;
import com.gbh.gbh_mm.asset.model.dto.DepositPaymentDto;
import com.gbh.gbh_mm.asset.model.dto.LoanListDto;
import com.gbh.gbh_mm.asset.model.dto.LoanPaymentDto;
import com.gbh.gbh_mm.asset.model.dto.SavingsListDto;
import com.gbh.gbh_mm.asset.model.dto.SavingsPaymentDto;
import com.gbh.gbh_mm.asset.model.dto.SavingsPaymentListDto;
import com.gbh.gbh_mm.asset.model.dto.WithdrawalAccountDto;
import com.gbh.gbh_mm.asset.model.entity.*;
import com.gbh.gbh_mm.asset.model.vo.request.RequestCheckAccountAuth;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindCardTransactionList;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindDepositDemandTransactionList;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindDepositPayment;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindLoanPaymentList;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindSavingsPaymentList;
import com.gbh.gbh_mm.asset.model.vo.request.RequestOpenAccountAuth;
import com.gbh.gbh_mm.asset.model.vo.request.RequestWithdrawalAccountTransfer;
import com.gbh.gbh_mm.asset.model.vo.request.RequestDeleteWithdrawalAccount;
import com.gbh.gbh_mm.asset.model.vo.response.*;
import com.gbh.gbh_mm.asset.repo.WithdrawalAccountRepository;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCreateAccountAuth;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindBilling;

import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestAccountTransfer;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.time.LocalDate;
import java.time.YearMonth;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindTransactionList;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindPayment;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFindRepaymentList;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsPayment;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.repo.UserRepository;
import jakarta.persistence.EntityNotFoundException;
import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.KeyGenerator;
import javax.crypto.NoSuchPaddingException;
import javax.crypto.SecretKey;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import lombok.AllArgsConstructor;
import org.modelmapper.ModelMapper;
import org.modelmapper.TypeToken;
import org.modelmapper.convention.MatchingStrategies;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class AssetServiceImpl implements AssetService {

    private final DemandDepositAPI demandDepositAPI;
    private final CardAPI cardAPI;
    private final LoanAPI loanAPI;
    private final SavingsAPI savingsAPI;
    private final DepositAPI depositAPI;
    private final AuthAPI authAPI;

    private final ModelMapper mapper;

    private final UserRepository userRepository;
    private final WithdrawalAccountRepository withdrawalAccountRepository;

    private final AlertService alertService;

//    @Override
//    public ResponseFindAssetList findAssetList(CustomUserDetails customUserDetails) {
//        mapper.getConfiguration().setMatchingStrategy(MatchingStrategies.STRICT);
//
//        ResponseFindAssetList response = new ResponseFindAssetList();
//
//        String userKey = customUserDetails.getUserKey();
//        String aesKey = customUserDetails.getAesKey();
//        byte[] decodedAes = Base64.getDecoder().decode(aesKey);
//        SecretKeySpec secretKeySpec = new SecretKeySpec(decodedAes, "AES");
//
//        try {
//            /* 목록 조회 API 호출 */
//            Map<String, Object> responseCardData =
//                cardAPI.findUserCardList(userKey);
//            Map<String, Object> responseDepositDemandData =
//                demandDepositAPI.findDemandDepositAccountList(userKey);
//            Map<String, Object> responseLoanData =
//                loanAPI.findAccountList(userKey);
//            Map<String, Object> responseSavingsData =
//                savingsAPI.findAccountList(userKey);
//            Map<String, Object> responseDepositData =
//                depositAPI.findAccountList(userKey);
//
//            Map<String, Object> cardApiData =
//                (Map<String, Object>) responseCardData.get("apiResponse");
//            Map<String, Object> depositDemandApiData =
//                (Map<String, Object>) responseDepositDemandData.get("apiResponse");
//            Map<String, Object> loanApiData =
//                (Map<String, Object>) responseLoanData.get("apiResponse");
//            Map<String, Object> savingsApiData =
//                (Map<String, Object>) responseSavingsData.get("apiResponse");
//            Map<String, Object> depositApiData =
//                (Map<String, Object>) responseDepositData.get("apiResponse");
//
//            List<Map<String, Object>> responseCardList =
//                (List<Map<String, Object>>) cardApiData.get("REC");
//            List<Map<String, Object>> responseDemandDepositList =
//                (List<Map<String, Object>>) depositDemandApiData.get("REC");
//            List<Map<String, Object>> responseLoanList =
//                (List<Map<String, Object>>) loanApiData.get("REC");
//            Map<String, Object> responseSavingsList =
//                (Map<String, Object>) savingsApiData.get("REC");
//            List<Map<String, Object>> savings =
//                (List<Map<String, Object>>) responseSavingsList.get("list");
//            Map<String, Object> responseDepositList =
//                (Map<String, Object>) depositApiData.get("REC");
//            List<Map<String, Object>> deposits =
//                (List<Map<String, Object>>) responseDepositList.get("list");
//
//            List<Card> cardList = responseCardList.stream()
//                .map(cardMap -> mapper.map(cardMap, Card.class))
//                .collect(Collectors.toList());
//
//            YearMonth current = YearMonth.now();
//            YearMonth lastMonth = current.minusMonths(1);
//
//            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMM");
//            String currentString = current.format(formatter);
//            String lastMonthString = lastMonth.format(formatter);
//
//            byte[] iv = new byte[16];
//            SecureRandom secureRandom = new SecureRandom();
//            secureRandom.nextBytes(iv);
//            IvParameterSpec ivParameterSpec = new IvParameterSpec(iv);
//
//            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
//            cipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, ivParameterSpec);
//
//            String encodedIV = Base64.getEncoder().encodeToString(iv);
//
//            long cardAmount = 0;
//            for (int i = 0; i < cardList.size(); i++) {
//                RequestFindBilling requestFindBilling = RequestFindBilling.builder()
//                    .cvc(cardList.get(i).getCvc())
//                    .cardNo(cardList.get(i).getCardNo())
//                    .startMonth(lastMonthString)
//                    .endMonth(currentString)
//                    .userKey(userKey)
//                    .build();
//
//                byte[] cardNoBytes = cipher
//                    .doFinal(cardList.get(i).getCardNo().getBytes(StandardCharsets.UTF_8));
//                byte[] cvcBytes = cipher
//                    .doFinal(cardList.get(i).getCvc().getBytes(StandardCharsets.UTF_8));
//
//                cardList.get(i).setCardNo(Base64.getEncoder().encodeToString(cardNoBytes));
//                cardList.get(i).setCvc(Base64.getEncoder().encodeToString(cvcBytes));
//
//                Map<String, Object> cardBillApi = cardAPI.findBilling(requestFindBilling);
//                Map<String, Object> cardBillData =
//                    (Map<String, Object>) cardBillApi.get("apiResponse");
//                List<Map<String, Object>> recList =
//                    (List<Map<String, Object>>) cardBillData.get("REC");
//
//                if (recList.size() > 0) {
//                    Map<String, Object> bill = recList.get(0);
//                    List<Map<String, Object>> billingList =
//                        (List<Map<String, Object>>) bill.get("billingList");
//                    Map<String, Object> billing = billingList.get(0);
//                    long totalAmount = Long.parseLong((String) billing.get("totalBalance"));
//
//                    cardAmount += totalAmount;
//
//                    String totalAmountString = String.valueOf(totalAmount);
//                    byte[] cardBalanceBytes = cipher
//                        .doFinal(totalAmountString.getBytes(StandardCharsets.UTF_8));
//                    cardList.get(i)
//                        .setCardBalance(Base64.getEncoder().encodeToString(cardBalanceBytes));
//                }
//            }
//
//            String totalAmountString = String.valueOf(cardAmount);
//            byte[] totalAmountBytes = cipher.doFinal(
//                totalAmountString.getBytes(StandardCharsets.UTF_8));
//
//            CardListDto responseCard = CardListDto.builder()
//                .totalAmount(Base64.getEncoder().encodeToString(totalAmountBytes))
//                .cardList(cardList)
//                .build();
//
//            /* 데이터 매핑(직렬화) */
//            List<DemandDeposit> demandDepositList = responseDemandDepositList.stream()
//                .map(demandDepositMap ->
//                    mapper.map(demandDepositMap, DemandDeposit.class))
//                .collect(Collectors.toList());
//
//            long demandDepositTotal = 0;
//            for (DemandDeposit demandDeposit : demandDepositList) {
//                long accountBalance = demandDeposit.getAccountBalance();
//                String accountBalanceString = String.valueOf(accountBalance);
//                demandDepositTotal += accountBalance;
//                byte[] accountNoBytes =
//                    cipher.doFinal(demandDeposit.getAccountNo().getBytes(StandardCharsets.UTF_8));
//                byte[] accountBalanceBytes =
//                    cipher.doFinal(accountBalanceString.getBytes(StandardCharsets.UTF_8));
//                demandDeposit.setAccountNo(Base64.getEncoder().encodeToString(accountNoBytes));
//                demandDeposit
//                    .setEncodedAccountBalance(
//                        Base64.getEncoder().encodeToString(accountBalanceBytes));
//                demandDeposit.setAccountBalance(0);
//            }
//
//            String demandDepositTotalString = String.valueOf(demandDepositTotal);
//            byte[] demandDepositTotalBytes =
//                cipher.doFinal(demandDepositTotalString.getBytes(StandardCharsets.UTF_8));
//
//            /* 총 금액 계산 */
//            DemandDepositListDto responseDemandDeposit = DemandDepositListDto.builder()
//                .demandDepositList(demandDepositList)
//                .totalAmount(Base64.getEncoder().encodeToString(demandDepositTotalBytes))
//                .build();
//
//            /* 데이터 매핑(직렬화) */
//            List<Loan> loanList = responseLoanList.stream()
//                .map(loanMap -> mapper.map(loanMap, Loan.class))
//                .collect(Collectors.toList());
//
//            /* 총 금액 계산 */
//            long loanTotalAmount = 0;
//
//            for (Loan loan : loanList) {
//                long loanBalance = loan.getLoanBalance();
//                String loanBalanceString = String.valueOf(loanBalance);
//                loanTotalAmount += loanBalance;
//
//                byte[] loanBalanceByte =
//                    cipher.doFinal(loanBalanceString.getBytes(StandardCharsets.UTF_8));
//                byte[] accountNoByte =
//                    cipher.doFinal(loan.getAccountNo().getBytes(StandardCharsets.UTF_8));
//
//                loan.setAccountNo(Base64.getEncoder().encodeToString(accountNoByte));
//                loan.setEncodeLoanBalance(Base64.getEncoder().encodeToString(loanBalanceByte));
//                loan.setLoanBalance(0);
//            }
//
//            String stringLoanTotal = String.valueOf(loanTotalAmount);
//            byte[] loanTotalBytes = cipher.doFinal(
//                stringLoanTotal.getBytes(StandardCharsets.UTF_8));
//
//            LoanListDto responseLoan = LoanListDto.builder()
//                .totalAmount(Base64.getEncoder().encodeToString(loanTotalBytes))
//                .loanList(loanList)
//                .build();
//
//            /* 데이터 매핑(직렬화) */
//            List<Savings> savingsList = savings.stream()
//                .map(savingsMap -> mapper.map(savingsMap, Savings.class))
//                .collect(Collectors.toList());
//
//            /* 총 금액 계산 */
//            long savingTotalAmount = 0;
//
//            for (Savings savings1 : savingsList) {
//                long savings1TotalBalance = savings1.getTotalBalance();
//                String savingsTotalBalanceString = String.valueOf(savings1TotalBalance);
//                savingTotalAmount += savings1TotalBalance;
//
//                byte[] byteTotalBalance =
//                    cipher.doFinal(savingsTotalBalanceString.getBytes(StandardCharsets.UTF_8));
//                byte[] accountNoByte =
//                    cipher.doFinal(savings1.getAccountNo().getBytes(StandardCharsets.UTF_8));
//
//                savings1.setAccountNo(Base64.getEncoder().encodeToString(accountNoByte));
//                savings1.setEncodedTotalBalance(
//                    Base64.getEncoder().encodeToString(byteTotalBalance));
//                savings1.setTotalBalance(0);
//            }
//
//            String savingsTotalString = String.valueOf(savingTotalAmount);
//            byte[] savingsTotalBytes =
//                cipher.doFinal(savingsTotalString.getBytes(StandardCharsets.UTF_8));
//
//            SavingsListDto responseSavings = SavingsListDto.builder()
//                .totalAmount(Base64.getEncoder().encodeToString(savingsTotalBytes))
//                .savingsList(savingsList)
//                .build();
//
//            /* 데이터 직렬화 */
//            List<Deposit> depositList = deposits.stream()
//                .map(depositMap -> mapper.map(depositMap, Deposit.class))
//                .collect(Collectors.toList());
//
//            /* 총 금액 계산 */
//            long depositTotalAmount = 0;
//
//            for (Deposit deposit : depositList) {
//                long depositBalance = deposit.getDepositBalance();
//                String depositBalanceString = String.valueOf(depositBalance);
//                depositTotalAmount += depositBalance;
//
//                byte[] depositBalanceByte =
//                    cipher.doFinal(depositBalanceString.getBytes(StandardCharsets.UTF_8));
//                byte[] accountNoByte =
//                    cipher.doFinal(deposit.getAccountNo().getBytes(StandardCharsets.UTF_8));
//                deposit.setAccountNo(Base64.getEncoder().encodeToString(accountNoByte));
//                deposit.setDepositBalance(0);
//                deposit.setEncodeDepositBalance(
//                    Base64.getEncoder().encodeToString(depositBalanceByte));
//            }
//
//            String depositTotalString = String.valueOf(depositTotalAmount);
//            byte[] depositTotalByte =
//                cipher.doFinal(depositTotalString.getBytes(StandardCharsets.UTF_8));
//
//            DepositListDto responseDeposit = DepositListDto.builder()
//                .totalAmount(Base64.getEncoder().encodeToString(depositTotalByte))
//                .depositList(depositList)
//                .build();
//
//            response.setCardData(responseCard);
//            response.setDemandDepositData(responseDemandDeposit);
//            response.setLoanData(responseLoan);
//            response.setSavingsData(responseSavings);
//            response.setDepositData(responseDeposit);
//            response.setIv(encodedIV);
//        } catch (JsonProcessingException e) {
//            throw new RuntimeException(e);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//
//        return response;
//    }

    @Override
    public ResponseFindAssetList findAssetList(CustomUserDetails customUserDetails) {
        mapper.getConfiguration().setMatchingStrategy(MatchingStrategies.STRICT);

        ResponseFindAssetList response = new ResponseFindAssetList();

        String userKey = customUserDetails.getUserKey();
        String aesKey = customUserDetails.getAesKey();
        byte[] decodedAes = Base64.getDecoder().decode(aesKey);
        SecretKeySpec secretKeySpec = new SecretKeySpec(decodedAes, "AES");

        try {
            /* 목록 조회 API 호출 */
            Map<String, Object> responseCardData =
                    cardAPI.findUserCardList(userKey);
            Map<String, Object> responseDepositDemandData =
                    demandDepositAPI.findDemandDepositAccountList(userKey);
            Map<String, Object> responseLoanData =
                    loanAPI.findAccountList(userKey);
            Map<String, Object> responseSavingsData =
                    savingsAPI.findAccountList(userKey);
            Map<String, Object> responseDepositData =
                    depositAPI.findAccountList(userKey);

            Map<String, Object> cardApiData =
                    (Map<String, Object>) responseCardData.get("apiResponse");
            Map<String, Object> depositDemandApiData =
                    (Map<String, Object>) responseDepositDemandData.get("apiResponse");
            Map<String, Object> loanApiData =
                    (Map<String, Object>) responseLoanData.get("apiResponse");
            Map<String, Object> savingsApiData =
                    (Map<String, Object>) responseSavingsData.get("apiResponse");
            Map<String, Object> depositApiData =
                    (Map<String, Object>) responseDepositData.get("apiResponse");

            List<Map<String, Object>> responseCardList =
                    (List<Map<String, Object>>) cardApiData.get("REC");
            List<Map<String, Object>> responseDemandDepositList =
                    (List<Map<String, Object>>) depositDemandApiData.get("REC");
            List<Map<String, Object>> responseLoanList =
                    (List<Map<String, Object>>) loanApiData.get("REC");
            Map<String, Object> responseSavingsList =
                    (Map<String, Object>) savingsApiData.get("REC");
            List<Map<String, Object>> savings =
                    (List<Map<String, Object>>) responseSavingsList.get("list");
            Map<String, Object> responseDepositList =
                    (Map<String, Object>) depositApiData.get("REC");
            List<Map<String, Object>> deposits =
                    (List<Map<String, Object>>) responseDepositList.get("list");

            List<Card> cardList = responseCardList.stream()
                    .map(cardMap -> mapper.map(cardMap, Card.class))
                    .collect(Collectors.toList());

            YearMonth current = YearMonth.now();
            YearMonth lastMonth = current.minusMonths(1);

            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMM");
            String currentString = current.format(formatter);
            String lastMonthString = lastMonth.format(formatter);

            byte[] iv = new byte[16];
            SecureRandom secureRandom = new SecureRandom();
            secureRandom.nextBytes(iv);
            IvParameterSpec ivParameterSpec = new IvParameterSpec(iv);

            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, ivParameterSpec);

            String encodedIV = Base64.getEncoder().encodeToString(iv);

            long cardAmount = 0;
            for (int i = 0; i < cardList.size(); i++) {
                com.gbh.gbh_mm.finance.card.vo.request.RequestFindCardTransactionList requestFindCardTransactionList =
                        com.gbh.gbh_mm.finance.card.vo.request.RequestFindCardTransactionList.builder()
                                .cardNo(cardList.get(i).getCardNo())
                                .userKey(userKey)
                                .cvc(cardList.get(i).getCvc())
                                .endDate("20251230")
                                .startDate("20000101")
                                .build();

                byte[] cardNoBytes = cipher
                        .doFinal(cardList.get(i).getCardNo().getBytes(StandardCharsets.UTF_8));
                byte[] cvcBytes = cipher
                        .doFinal(cardList.get(i).getCvc().getBytes(StandardCharsets.UTF_8));

                cardList.get(i).setCardNo(Base64.getEncoder().encodeToString(cardNoBytes));
                cardList.get(i).setCvc(Base64.getEncoder().encodeToString(cvcBytes));

                Map<String, Object> cardTransactionResponseData = cardAPI.findTransactionList(requestFindCardTransactionList);
                Map<String, Object> cardTransactionApiData =
                        (Map<String, Object>) cardTransactionResponseData.get("apiResponse");
                Map<String, Object> cardRecData = (Map<String, Object>) cardTransactionApiData.get("REC");

                String estimatedBalance = (String) cardRecData.get("estimatedBalance");

                if (!estimatedBalance.isEmpty() && estimatedBalance != null) {
                    byte[] estimatedBalanceByte = cipher.doFinal(estimatedBalance.getBytes(StandardCharsets.UTF_8));
                        cardList.get(i).setCardBalance(Base64.getEncoder().encodeToString(estimatedBalanceByte));
                }

            }

            String totalAmountString = String.valueOf(cardAmount);
            byte[] totalAmountBytes = cipher.doFinal(
                    totalAmountString.getBytes(StandardCharsets.UTF_8));

            CardListDto responseCard = CardListDto.builder()
                    .totalAmount(Base64.getEncoder().encodeToString(totalAmountBytes))
                    .cardList(cardList)
                    .build();

            /* 데이터 매핑(직렬화) */
            List<DemandDeposit> demandDepositList = responseDemandDepositList.stream()
                    .map(demandDepositMap ->
                            mapper.map(demandDepositMap, DemandDeposit.class))
                    .collect(Collectors.toList());

            long demandDepositTotal = 0;
            for (DemandDeposit demandDeposit : demandDepositList) {
                long accountBalance = demandDeposit.getAccountBalance();
                String accountBalanceString = String.valueOf(accountBalance);
                demandDepositTotal += accountBalance;
                byte[] accountNoBytes =
                        cipher.doFinal(demandDeposit.getAccountNo().getBytes(StandardCharsets.UTF_8));
                byte[] accountBalanceBytes =
                        cipher.doFinal(accountBalanceString.getBytes(StandardCharsets.UTF_8));
                demandDeposit.setAccountNo(Base64.getEncoder().encodeToString(accountNoBytes));
                demandDeposit
                        .setEncodedAccountBalance(
                                Base64.getEncoder().encodeToString(accountBalanceBytes));
                demandDeposit.setAccountBalance(0);
            }

            String demandDepositTotalString = String.valueOf(demandDepositTotal);
            byte[] demandDepositTotalBytes =
                    cipher.doFinal(demandDepositTotalString.getBytes(StandardCharsets.UTF_8));

            /* 총 금액 계산 */
            DemandDepositListDto responseDemandDeposit = DemandDepositListDto.builder()
                    .demandDepositList(demandDepositList)
                    .totalAmount(Base64.getEncoder().encodeToString(demandDepositTotalBytes))
                    .build();

            /* 데이터 매핑(직렬화) */
            List<Loan> loanList = responseLoanList.stream()
                    .map(loanMap -> mapper.map(loanMap, Loan.class))
                    .collect(Collectors.toList());

            /* 총 금액 계산 */
            long loanTotalAmount = 0;

            for (Loan loan : loanList) {
                long loanBalance = loan.getLoanBalance();
                String loanBalanceString = String.valueOf(loanBalance);
                loanTotalAmount += loanBalance;

                byte[] loanBalanceByte =
                        cipher.doFinal(loanBalanceString.getBytes(StandardCharsets.UTF_8));
                byte[] accountNoByte =
                        cipher.doFinal(loan.getAccountNo().getBytes(StandardCharsets.UTF_8));

                loan.setAccountNo(Base64.getEncoder().encodeToString(accountNoByte));
                loan.setEncodeLoanBalance(Base64.getEncoder().encodeToString(loanBalanceByte));
                loan.setLoanBalance(0);
            }

            String stringLoanTotal = String.valueOf(loanTotalAmount);
            byte[] loanTotalBytes = cipher.doFinal(
                    stringLoanTotal.getBytes(StandardCharsets.UTF_8));

            LoanListDto responseLoan = LoanListDto.builder()
                    .totalAmount(Base64.getEncoder().encodeToString(loanTotalBytes))
                    .loanList(loanList)
                    .build();

            /* 데이터 매핑(직렬화) */
            List<Savings> savingsList = savings.stream()
                    .map(savingsMap -> mapper.map(savingsMap, Savings.class))
                    .collect(Collectors.toList());

            /* 총 금액 계산 */
            long savingTotalAmount = 0;

            for (Savings savings1 : savingsList) {
                long savings1TotalBalance = savings1.getTotalBalance();
                String savingsTotalBalanceString = String.valueOf(savings1TotalBalance);
                savingTotalAmount += savings1TotalBalance;

                byte[] byteTotalBalance =
                        cipher.doFinal(savingsTotalBalanceString.getBytes(StandardCharsets.UTF_8));
                byte[] accountNoByte =
                        cipher.doFinal(savings1.getAccountNo().getBytes(StandardCharsets.UTF_8));

                savings1.setAccountNo(Base64.getEncoder().encodeToString(accountNoByte));
                savings1.setEncodedTotalBalance(
                        Base64.getEncoder().encodeToString(byteTotalBalance));
                savings1.setTotalBalance(0);
            }

            String savingsTotalString = String.valueOf(savingTotalAmount);
            byte[] savingsTotalBytes =
                    cipher.doFinal(savingsTotalString.getBytes(StandardCharsets.UTF_8));

            SavingsListDto responseSavings = SavingsListDto.builder()
                    .totalAmount(Base64.getEncoder().encodeToString(savingsTotalBytes))
                    .savingsList(savingsList)
                    .build();

            /* 데이터 직렬화 */
            List<Deposit> depositList = deposits.stream()
                    .map(depositMap -> mapper.map(depositMap, Deposit.class))
                    .collect(Collectors.toList());

            /* 총 금액 계산 */
            long depositTotalAmount = 0;

            for (Deposit deposit : depositList) {
                long depositBalance = deposit.getDepositBalance();
                String depositBalanceString = String.valueOf(depositBalance);
                depositTotalAmount += depositBalance;

                byte[] depositBalanceByte =
                        cipher.doFinal(depositBalanceString.getBytes(StandardCharsets.UTF_8));
                byte[] accountNoByte =
                        cipher.doFinal(deposit.getAccountNo().getBytes(StandardCharsets.UTF_8));
                deposit.setAccountNo(Base64.getEncoder().encodeToString(accountNoByte));
                deposit.setDepositBalance(0);
                deposit.setEncodeDepositBalance(
                        Base64.getEncoder().encodeToString(depositBalanceByte));
            }

            String depositTotalString = String.valueOf(depositTotalAmount);
            byte[] depositTotalByte =
                    cipher.doFinal(depositTotalString.getBytes(StandardCharsets.UTF_8));

            DepositListDto responseDeposit = DepositListDto.builder()
                    .totalAmount(Base64.getEncoder().encodeToString(depositTotalByte))
                    .depositList(depositList)
                    .build();

            response.setCardData(responseCard);
            response.setDemandDepositData(responseDemandDeposit);
            response.setLoanData(responseLoan);
            response.setSavingsData(responseSavings);
            response.setDepositData(responseDeposit);
            response.setIv(encodedIV);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return response;
    }

    @Override
    public ResponseFindAssetList findAssetListWithNoEncrypt(CustomUserDetails customUserDetails) {
        mapper.getConfiguration().setMatchingStrategy(MatchingStrategies.STRICT);

        ResponseFindAssetList response = new ResponseFindAssetList();

        String userKey = customUserDetails.getUserKey();
        String aesKey = customUserDetails.getAesKey();

        try {
            /* 목록 조회 API 호출 */
            Map<String, Object> responseCardData =
                    cardAPI.findUserCardList(userKey);
            Map<String, Object> responseDepositDemandData =
                    demandDepositAPI.findDemandDepositAccountList(userKey);
            Map<String, Object> responseLoanData =
                    loanAPI.findAccountList(userKey);
            Map<String, Object> responseSavingsData =
                    savingsAPI.findAccountList(userKey);
            Map<String, Object> responseDepositData =
                    depositAPI.findAccountList(userKey);

            Map<String, Object> cardApiData =
                    (Map<String, Object>) responseCardData.get("apiResponse");
            Map<String, Object> depositDemandApiData =
                    (Map<String, Object>) responseDepositDemandData.get("apiResponse");
            Map<String, Object> loanApiData =
                    (Map<String, Object>) responseLoanData.get("apiResponse");
            Map<String, Object> savingsApiData =
                    (Map<String, Object>) responseSavingsData.get("apiResponse");
            Map<String, Object> depositApiData =
                    (Map<String, Object>) responseDepositData.get("apiResponse");

            List<Map<String, Object>> responseCardList =
                    (List<Map<String, Object>>) cardApiData.get("REC");
            List<Map<String, Object>> responseDemandDepositList =
                    (List<Map<String, Object>>) depositDemandApiData.get("REC");
            List<Map<String, Object>> responseLoanList =
                    (List<Map<String, Object>>) loanApiData.get("REC");
            Map<String, Object> responseSavingsList =
                    (Map<String, Object>) savingsApiData.get("REC");
            List<Map<String, Object>> savings =
                    (List<Map<String, Object>>) responseSavingsList.get("list");
            Map<String, Object> responseDepositList =
                    (Map<String, Object>) depositApiData.get("REC");
            List<Map<String, Object>> deposits =
                    (List<Map<String, Object>>) responseDepositList.get("list");

            List<Card> cardList = responseCardList.stream()
                    .map(cardMap -> mapper.map(cardMap, Card.class))
                    .collect(Collectors.toList());

            YearMonth current = YearMonth.now();
            YearMonth lastMonth = current.minusMonths(1);

            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMM");
            String currentString = current.format(formatter);
            String lastMonthString = lastMonth.format(formatter);

            long cardAmount = 0;
            for (Card card : cardList) {
                RequestFindBilling requestFindBilling = RequestFindBilling.builder()
                        .cvc(card.getCvc())
                        .cardNo(card.getCardNo())
                        .startMonth(lastMonthString)
                        .endMonth(currentString)
                        .userKey(userKey)
                        .build();

                Map<String, Object> cardBillApi = cardAPI.findBilling(requestFindBilling);
                Map<String, Object> cardBillData =
                        (Map<String, Object>) cardBillApi.get("apiResponse");
                List<Map<String, Object>> recList =
                        (List<Map<String, Object>>) cardBillData.get("REC");

                if (!recList.isEmpty()) {
                    Map<String, Object> bill = recList.getFirst();
                    List<Map<String, Object>> billingList =
                            (List<Map<String, Object>>) bill.get("billingList");
                    Map<String, Object> billing = billingList.getFirst();
                    long totalAmount = Long.parseLong((String) billing.get("totalBalance"));

                    cardAmount += totalAmount;

                    String totalAmountString = String.valueOf(totalAmount);
                    card.setCardBalance(String.valueOf(totalAmount));
                }
            }

            CardListDto responseCard = CardListDto.builder()
                    .totalAmount(String.valueOf(cardAmount))
                    .cardList(cardList)
                    .build();

            /* 데이터 매핑(직렬화) */
            List<DemandDeposit> demandDepositList = responseDemandDepositList.stream()
                    .map(demandDepositMap ->
                            mapper.map(demandDepositMap, DemandDeposit.class))
                    .collect(Collectors.toList());

            long demandDepositTotal = 0;
            for (DemandDeposit demandDeposit : demandDepositList) {
                demandDepositTotal += demandDeposit.getAccountBalance();
                demandDeposit.setEncodedAccountBalance(String.valueOf(demandDeposit.getAccountBalance()));
            }

            /* 총 금액 계산 */
            DemandDepositListDto responseDemandDeposit = DemandDepositListDto.builder()
                    .demandDepositList(demandDepositList)
                    .totalAmount(String.valueOf(demandDepositTotal))
                    .build();

            /* 데이터 매핑(직렬화) */
            List<Loan> loanList = responseLoanList.stream()
                    .map(loanMap -> mapper.map(loanMap, Loan.class))
                    .collect(Collectors.toList());

            /* 총 금액 계산 */
            long loanTotalAmount = 0;

            for (Loan loan : loanList) {
                loanTotalAmount += loan.getLoanBalance();
                loan.setEncodeLoanBalance(String.valueOf(loan.getLoanBalance()));
            }

            LoanListDto responseLoan = LoanListDto.builder()
                    .totalAmount(String.valueOf(loanTotalAmount))
                    .loanList(loanList)
                    .build();

            List<Savings> savingsList = savings.stream()
                    .map(map -> mapper.map(map, Savings.class))
                    .collect(Collectors.toList());

            long savingsTotal = 0;
            for (Savings saving : savingsList) {
                savingsTotal += saving.getTotalBalance();
                saving.setEncodedTotalBalance(String.valueOf(saving.getTotalBalance()));
            }

            SavingsListDto responseSavings = SavingsListDto.builder()
                    .totalAmount(String.valueOf(savingsTotal))
                    .savingsList(savingsList)
                    .build();

            List<Deposit> depositList = deposits.stream()
                    .map(map -> mapper.map(map, Deposit.class))
                    .collect(Collectors.toList());

            long depositTotal = 0;
            for (Deposit deposit : depositList) {
                depositTotal += deposit.getDepositBalance();
                deposit.setEncodeDepositBalance(String.valueOf(deposit.getDepositBalance()));
            }

            DepositListDto responseDeposit = DepositListDto.builder()
                    .totalAmount(String.valueOf(depositTotal))
                    .depositList(depositList)
                    .build();

            response.setCardData(responseCard);
            response.setDemandDepositData(responseDemandDeposit);
            response.setLoanData(responseLoan);
            response.setSavingsData(responseSavings);
            response.setDepositData(responseDeposit);
            response.setIv(null); // IV 필요 없음
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return response;
    }

    @Override
    public ResponseFindDepositDemandTransactionList findDepositDemandTransactionList(
        RequestFindDepositDemandTransactionList request,
        CustomUserDetails customUserDetails
    ) {
        ResponseFindDepositDemandTransactionList response =
            new ResponseFindDepositDemandTransactionList();
        try {
            String encodedKey = customUserDetails.getAesKey();
            String encodedIv = request.getIv();

            byte[] decodedKey = Base64.getDecoder().decode(encodedKey);
            byte[] decodedIv = Base64.getDecoder().decode(encodedIv);
            byte[] decodedAccountNo = Base64.getDecoder().decode(request.getAccountNo());

            SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");
            IvParameterSpec ivParameterSpec = new IvParameterSpec(decodedIv);

            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec);

            byte[] decryptedByte = cipher.doFinal(decodedAccountNo);
            String accountNo = new String(decryptedByte, StandardCharsets.UTF_8);

            RequestFindTransactionList requestApi = RequestFindTransactionList.builder()
                .accountNo(accountNo)
                .startDate(request.getStartDate())
                .endDate(request.getEndDate())
                .transactionType(request.getTransactionType())
                .orderByType(request.getOrderByType())
                .userKey(customUserDetails.getUserKey())
                .build();

            Map<String, Object> apiData = demandDepositAPI.findTransactionList(requestApi);
            Map<String, Object> responseData = (Map<String, Object>) apiData.get("apiResponse");
            Map<String, Object> transactionData = (Map<String, Object>) responseData.get("REC");
            List<Map<String, Object>> transactionList =
                (List<Map<String, Object>>) transactionData.get("list");

            byte[] newIv = new byte[16];
            SecureRandom secureRandom = new SecureRandom();
            secureRandom.nextBytes(newIv);
            IvParameterSpec newIvParameterSpec = new IvParameterSpec(newIv);

            Cipher newCipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            newCipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, newIvParameterSpec);

            String encodedNewIv = Base64.getEncoder().encodeToString(newIv);

            List<DemandDepositTransactionDto> demandDepositTransactionDtos = new ArrayList<>();

            for (Map<String, Object> transaction : transactionList) {
                String transactionUniqueNo = (String) transaction.get("transactionUniqueNo");
                byte[] byteTransactionUniqueNo =
                    newCipher.doFinal(transactionUniqueNo.getBytes(StandardCharsets.UTF_8));
                String transactionDate = (String) transaction.get("transactionDate");
                byte[] byteTransactionDate =
                    newCipher.doFinal(transactionDate.getBytes(StandardCharsets.UTF_8));
                String transactionTime = (String) transaction.get("transactionTime");
                byte[] byteTransactionTime =
                    newCipher.doFinal(transactionTime.getBytes(StandardCharsets.UTF_8));
                String transactionType = (String) transaction.get("transactionType");
                byte[] byteTransactionType =
                    newCipher.doFinal(transactionType.getBytes(StandardCharsets.UTF_8));
                String transactionTypeName = (String) transaction.get("transactionTypeName");
                byte[] byteTransactionTypeName =
                    newCipher.doFinal(transactionTypeName.getBytes(StandardCharsets.UTF_8));
                String transactionAccountNo = (String) transaction.get("transactionAccountNo");
                byte[] byteTransactionAccountNo =
                    newCipher.doFinal(transactionAccountNo.getBytes(StandardCharsets.UTF_8));
                String transactionBalance = (String) transaction.get("transactionBalance");
                byte[] byteTransactionBalance =
                    newCipher.doFinal(transactionBalance.getBytes(StandardCharsets.UTF_8));
                String transactionAfterBalance = (String) transaction.get(
                    "transactionAfterBalance");
                byte[] byteTransactionAfterBalance =
                    newCipher.doFinal(transactionAfterBalance.getBytes(StandardCharsets.UTF_8));
                String transactionSummary = (String) transaction.get("transactionSummary");
                byte[] byteTransactionSummary =
                    newCipher.doFinal(transactionSummary.getBytes(StandardCharsets.UTF_8));
                String transactionMemo = (String) transaction.get("transactionMemo");
                byte[] byteTransactionTypeMemo =
                    newCipher.doFinal(transactionMemo.getBytes(StandardCharsets.UTF_8));

                DemandDepositTransactionDto demandDepositTransactionDto =
                    DemandDepositTransactionDto.builder()
                        .transactionUniqueNo(
                            Base64.getEncoder().encodeToString(byteTransactionUniqueNo))
                        .transactionDate(Base64.getEncoder().encodeToString(byteTransactionDate))
                        .transactionTime(Base64.getEncoder().encodeToString(byteTransactionTime))
                        .transactionType(Base64.getEncoder().encodeToString(byteTransactionType))
                        .transactionTypeName(
                            Base64.getEncoder().encodeToString(byteTransactionTypeName))
                        .transactionAccountNo(
                            Base64.getEncoder().encodeToString(byteTransactionAccountNo))
                        .transactionBalance(
                            Base64.getEncoder().encodeToString(byteTransactionBalance))
                        .transactionAfterBalance(
                            Base64.getEncoder().encodeToString(byteTransactionAfterBalance))
                        .transactionSummary(
                            Base64.getEncoder().encodeToString(byteTransactionSummary))
                        .transactionMemo(
                            Base64.getEncoder().encodeToString(byteTransactionTypeMemo))
                        .build();

                demandDepositTransactionDtos.add(demandDepositTransactionDto);
            }

            response.setTransactionList(demandDepositTransactionDtos);
            response.setIv(encodedNewIv);

        } catch (Exception e) {
            e.printStackTrace();
        }
        return response;
    }

    @Override
    public ResponseFindDepositPayment findDepositPayment(
        RequestFindDepositPayment request,
        CustomUserDetails customUserDetails
    ) {
        ResponseFindDepositPayment response = new ResponseFindDepositPayment();

        try {
            String encodedKey = customUserDetails.getAesKey();
            String encodedIv = request.getIv();

            byte[] decodedKey = Base64.getDecoder().decode(encodedKey);
            byte[] decodedIv = Base64.getDecoder().decode(encodedIv);
            byte[] decodedAccountNo = Base64.getDecoder().decode(request.getAccountNo());

            SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");
            IvParameterSpec ivParameterSpec = new IvParameterSpec(decodedIv);

            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec);

            byte[] decryptedByte = cipher.doFinal(decodedAccountNo);
            String accountNo = new String(decryptedByte, StandardCharsets.UTF_8);

            RequestFindPayment requestApi = RequestFindPayment.builder()
                .accountNo(accountNo)
                .userKey(customUserDetails.getUserKey())
                .build();
            Map<String, Object> apiData = depositAPI.findPayment(requestApi);
            Map<String, Object> responseData = (Map<String, Object>) apiData.get("apiResponse");
            Map<String, Object> paymentData = (Map<String, Object>) responseData.get("REC");

            byte[] newIv = new byte[16];
            SecureRandom secureRandom = new SecureRandom();
            secureRandom.nextBytes(newIv);
            IvParameterSpec newIvParameterSpec = new IvParameterSpec(newIv);

            Cipher newCipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            newCipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, newIvParameterSpec);

            String encodedNewIv = Base64.getEncoder().encodeToString(newIv);

            String paymentUniqueNo = (String) paymentData.get("paymentUniqueNo");
            byte[] paymentUniqueNoByte =
                newCipher.doFinal(paymentUniqueNo.getBytes(StandardCharsets.UTF_8));
            String paymentDate = (String) paymentData.get("paymentDate");
            byte[] paymentDateByte =
                newCipher.doFinal(paymentDate.getBytes(StandardCharsets.UTF_8));
            String paymentTime = (String) paymentData.get("paymentTime");
            byte[] paymentTimeByte =
                newCipher.doFinal(paymentTime.getBytes(StandardCharsets.UTF_8));
            String paymentBalance = (String) paymentData.get("paymentBalance");
            byte[] paymentBalanceByte =
                newCipher.doFinal(paymentBalance.getBytes(StandardCharsets.UTF_8));

            DepositPaymentDto depositPaymentDto = DepositPaymentDto.builder()
                .paymentUniqueNo(Base64.getEncoder().encodeToString(paymentUniqueNoByte))
                .paymentDate(Base64.getEncoder().encodeToString(paymentDateByte))
                .paymentTime(Base64.getEncoder().encodeToString(paymentTimeByte))
                .paymentBalance(Base64.getEncoder().encodeToString(paymentBalanceByte))
                .build();

            response.setPayment(depositPaymentDto);
            response.setIv(encodedNewIv);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return response;
    }

    @Override
    public ResponseFindSavingsPaymentList findSavingsPaymentList(
        RequestFindSavingsPaymentList request,
        CustomUserDetails customUserDetails
    ) {
        ResponseFindSavingsPaymentList response = new ResponseFindSavingsPaymentList();

        try {
            String encodedKey = customUserDetails.getAesKey();
            String encodedIv = request.getIv();

            byte[] decodedKey = Base64.getDecoder().decode(encodedKey);
            byte[] decodedIv = Base64.getDecoder().decode(encodedIv);
            byte[] decodedAccountNo = Base64.getDecoder().decode(request.getAccountNo());

            SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");
            IvParameterSpec ivParameterSpec = new IvParameterSpec(decodedIv);

            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec);

            byte[] decryptedByte = cipher.doFinal(decodedAccountNo);
            String accountNo = new String(decryptedByte, StandardCharsets.UTF_8);

            RequestFindSavingsPayment requestApi = RequestFindSavingsPayment.builder()
                .accountNo(accountNo)
                .userKey(customUserDetails.getUserKey())
                .build();
            Map<String, Object> apiData = savingsAPI.findPayment(requestApi);
            Map<String, Object> responseData = (Map<String, Object>) apiData.get("apiResponse");
            List<Map<String, Object>> paymentList = (List<Map<String, Object>>) responseData.get(
                "REC");

            byte[] newIv = new byte[16];
            SecureRandom secureRandom = new SecureRandom();
            secureRandom.nextBytes(newIv);
            IvParameterSpec newIvParameterSpec = new IvParameterSpec(newIv);

            Cipher newCipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            newCipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, newIvParameterSpec);

            String encodedNewIv = Base64.getEncoder().encodeToString(newIv);

            Map<String, Object> paymentInfo = paymentList.get(0);
            List<Map<String, Object>> savingsPaymentList =
                (List<Map<String, Object>>) paymentInfo.get("paymentInfo");

            List<SavingsPaymentListDto> savingsPaymentListDtos = new ArrayList<>();
            for (Map<String, Object> stringObjectMap : savingsPaymentList) {
                String depositInstallment = (String) stringObjectMap.get("depositInstallment");
                byte[] depositInstallmentByte =
                    newCipher.doFinal(depositInstallment.getBytes(StandardCharsets.UTF_8));
                String paymentBalance = (String) stringObjectMap.get("paymentBalance");
                byte[] paymentBalanceByte =
                    newCipher.doFinal(paymentBalance.getBytes(StandardCharsets.UTF_8));
                String paymentDate = (String) stringObjectMap.get("paymentDate");
                byte[] paymentDateByte =
                    newCipher.doFinal(paymentDate.getBytes(StandardCharsets.UTF_8));
                String paymentTime = (String) stringObjectMap.get("paymentTime");
                byte[] paymentTimeByte =
                    newCipher.doFinal(paymentTime.getBytes(StandardCharsets.UTF_8));
                String status = (String) stringObjectMap.get("status");
                byte[] statusByte =
                    newCipher.doFinal(status.getBytes(StandardCharsets.UTF_8));

                SavingsPaymentListDto savingsPaymentListDto = SavingsPaymentListDto.builder()
                    .depositInstallment(Base64.getEncoder().encodeToString(depositInstallmentByte))
                    .paymentBalance(Base64.getEncoder().encodeToString(paymentBalanceByte))
                    .paymentDate(Base64.getEncoder().encodeToString(paymentDateByte))
                    .paymentTime(Base64.getEncoder().encodeToString(paymentTimeByte))
                    .status(Base64.getEncoder().encodeToString(statusByte))
                    .build();

                if (stringObjectMap.get("failureReason") != null) {
                    String failureReason = (String) stringObjectMap.get("failureReason");
                    byte[] failureReasonByte =
                        newCipher.doFinal(failureReason.getBytes(StandardCharsets.UTF_8));
                    savingsPaymentListDto
                        .setFailureReason(Base64.getEncoder().encodeToString(failureReasonByte));
                }

                savingsPaymentListDtos.add(savingsPaymentListDto);
            }

            String savingsAccountNo = (String) paymentInfo.get("accountNo");
            byte[] savingsAccountNoByte =
                newCipher.doFinal(savingsAccountNo.getBytes(StandardCharsets.UTF_8));
            String depositBalance = (String) paymentInfo.get("depositBalance");
            byte[] depositBalanceByte =
                newCipher.doFinal(depositBalance.getBytes(StandardCharsets.UTF_8));
            String totalBalance = (String) paymentInfo.get("totalBalance");
            byte[] totalBalanceByte =
                newCipher.doFinal(totalBalance.getBytes(StandardCharsets.UTF_8));
            String accountCreateDate = (String) paymentInfo.get("accountCreateDate");
            byte[] accountCreateDateByte =
                newCipher.doFinal(accountCreateDate.getBytes(StandardCharsets.UTF_8));
            String accountExpireDate = (String) paymentInfo.get("accountExpiryDate");
            byte[] accountExpireDateByte =
                newCipher.doFinal(accountExpireDate.getBytes(StandardCharsets.UTF_8));

            SavingsPaymentDto savingsPaymentDto = SavingsPaymentDto.builder()
                .bankCode((String) paymentInfo.get("bankCode"))
                .bankName((String) paymentInfo.get("bankName"))
                .accountNo(Base64.getEncoder().encodeToString(savingsAccountNoByte))
                .interestRate((String) paymentInfo.get("interestRate"))
                .depositBalance(Base64.getEncoder().encodeToString(depositBalanceByte))
                .totalBalance(Base64.getEncoder().encodeToString(totalBalanceByte))
                .accountCreateDate(Base64.getEncoder().encodeToString(accountCreateDateByte))
                .accountExpiryDate(Base64.getEncoder().encodeToString(accountExpireDateByte))
                .paymentInfo(savingsPaymentListDtos)
                .build();

            response.setPaymentList(savingsPaymentDto);
            response.setIv(encodedNewIv);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return response;
    }

    @Override
    public ResponseFindLoanPaymentList findLoanPaymentList(
        RequestFindLoanPaymentList request,
        CustomUserDetails customUserDetails
    ) {
        mapper.getConfiguration().setMatchingStrategy(MatchingStrategies.STRICT);

        try {
            String encodedKey = customUserDetails.getAesKey();
            String encodedIv = request.getIv();

            byte[] decodedKey = Base64.getDecoder().decode(encodedKey);
            byte[] decodedIv = Base64.getDecoder().decode(encodedIv);
            byte[] decodedAccountNo = Base64.getDecoder().decode(request.getAccountNo());

            SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");
            IvParameterSpec ivParameterSpec = new IvParameterSpec(decodedIv);

            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec);

            byte[] decryptedByte = cipher.doFinal(decodedAccountNo);
            String accountNo = new String(decryptedByte, StandardCharsets.UTF_8);

            RequestFindRepaymentList requestApi = RequestFindRepaymentList.builder()
                .accountNo(accountNo)
                .userKey(customUserDetails.getUserKey())
                .build();
            Map<String, Object> apiData = loanAPI.findRepaymentList(requestApi);
            Map<String, Object> responseData = (Map<String, Object>) apiData.get("apiResponse");
            Map<String, Object> recData = (Map<String, Object>) responseData.get("REC");
            List<Map<String, Object>> repaymentRecords =
                (List<Map<String, Object>>) recData.get("repaymentRecords");

            byte[] newIv = new byte[16];
            SecureRandom secureRandom = new SecureRandom();
            secureRandom.nextBytes(newIv);
            IvParameterSpec newIvParameterSpec = new IvParameterSpec(newIv);

            Cipher newCipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            newCipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, newIvParameterSpec);

            String encodedNewIv = Base64.getEncoder().encodeToString(newIv);

            List<LoanPaymentDto> loanPaymentDtoList = new ArrayList<>();
            for (Map<String, Object> repaymentRecord : repaymentRecords) {
                String installmentNumber = (String) repaymentRecord.get("installmentNumber");
                byte[] installmentNumberByte =
                    newCipher.doFinal(installmentNumber.getBytes(StandardCharsets.UTF_8));
                String status = (String) repaymentRecord.get("status");
                byte[] statusByte = newCipher.doFinal(status.getBytes(StandardCharsets.UTF_8));
                String paymentBalance = (String) repaymentRecord.get("paymentBalance");
                byte[] paymentBalanceByte =
                    newCipher.doFinal(paymentBalance.getBytes(StandardCharsets.UTF_8));
                String repaymentAttemptDate = (String) repaymentRecord.get("repaymentAttemptDate");
                byte[] repaymentAttemptDateByte =
                    newCipher.doFinal(repaymentAttemptDate.getBytes(StandardCharsets.UTF_8));
                String repaymentAttemptTime = (String) repaymentRecord.get("repaymentAttemptTime");
                byte[] repaymentAttemptTimeByte =
                    newCipher.doFinal(repaymentAttemptTime.getBytes(StandardCharsets.UTF_8));
                String repaymentActualDate = (String) repaymentRecord.get("repaymentActualDate");
                byte[] repaymentActualDateByte =
                    newCipher.doFinal(repaymentActualDate.getBytes(StandardCharsets.UTF_8));
                String repaymentActualTime = (String) repaymentRecord.get("repaymentActualTime");
                byte[] repaymentActualTimeByte =
                    newCipher.doFinal(repaymentActualTime.getBytes(StandardCharsets.UTF_8));
                String failureReason = (String) repaymentRecord.get("failureReason");
                byte[] failureReasonByte =
                    newCipher.doFinal(failureReason.getBytes(StandardCharsets.UTF_8));

                LoanPaymentDto loanPaymentDto = LoanPaymentDto.builder()
                    .installmentNumber(Base64.getEncoder().encodeToString(installmentNumberByte))
                    .status(Base64.getEncoder().encodeToString(statusByte))
                    .paymentBalance(Base64.getEncoder().encodeToString(paymentBalanceByte))
                    .repaymentAttemptDate(
                        Base64.getEncoder().encodeToString(repaymentAttemptDateByte))
                    .repaymentAttemptTime(
                        Base64.getEncoder().encodeToString(repaymentAttemptTimeByte))
                    .repaymentActualDate(
                        Base64.getEncoder().encodeToString(repaymentActualDateByte))
                    .repaymentActualTime(
                        Base64.getEncoder().encodeToString(repaymentActualTimeByte))
                    .failureReason(Base64.getEncoder().encodeToString(failureReasonByte))
                    .build();

                loanPaymentDtoList.add(loanPaymentDto);
            }

            String accountStatus = (String) recData.get("status");
            byte[] accountStatusByte =
                newCipher.doFinal(accountStatus.getBytes(StandardCharsets.UTF_8));
            String loanBalance = (String) recData.get("loanBalance");
            byte[] loanBalanceByte =
                newCipher.doFinal(loanBalance.getBytes(StandardCharsets.UTF_8));
            String remainingLoanBalance = (String) recData.get("remainingLoanBalance");
            byte[] remainingLoanBalanceByte =
                newCipher.doFinal(remainingLoanBalance.getBytes(StandardCharsets.UTF_8));

            ResponseFindLoanPaymentList response = ResponseFindLoanPaymentList.builder()
                .iv(encodedNewIv)
                .status(Base64.getEncoder().encodeToString(accountStatusByte))
                .loanBalance(Base64.getEncoder().encodeToString(loanBalanceByte))
                .remainingLoanBalance(Base64.getEncoder().encodeToString(remainingLoanBalanceByte))
                .repaymentRecords(loanPaymentDtoList)
                .build();

            return response;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public ResponseFindCardTransactionList findCardTransactionList(
        RequestFindCardTransactionList request,
        CustomUserDetails customUserDetails
    ) {
        try {
            String encodedKey = customUserDetails.getAesKey();
            String encodedIv = request.getIv();

            byte[] decodedKey = Base64.getDecoder().decode(encodedKey);
            byte[] decodedIv = Base64.getDecoder().decode(encodedIv);
            byte[] decodedCardNo = Base64.getDecoder().decode(request.getCardNo());
            byte[] decodedCvc = Base64.getDecoder().decode(request.getCvc());

            SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");
            IvParameterSpec ivParameterSpec = new IvParameterSpec(decodedIv);

            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec);

            byte[] decryptedCardNo = cipher.doFinal(decodedCardNo);
            byte[] decryptedCvc = cipher.doFinal(decodedCvc);
            String cardNo = new String(decryptedCardNo, StandardCharsets.UTF_8);
            String cvc = new String(decryptedCvc, StandardCharsets.UTF_8);

            com.gbh.gbh_mm.finance.card.vo.request.RequestFindCardTransactionList requestApi =
                com.gbh.gbh_mm.finance.card.vo.request.RequestFindCardTransactionList.builder()
                    .cardNo(cardNo)
                    .cvc(cvc)
                    .startDate(request.getStartDate())
                    .endDate(request.getEndDate())
                    .userKey(customUserDetails.getUserKey())
                    .build();

            Map<String, Object> apiData = cardAPI.findTransactionList(requestApi);
            Map<String, Object> responseData = (Map<String, Object>) apiData.get("apiResponse");
            Map<String, Object> recData = (Map<String, Object>) responseData.get("REC");

            byte[] newIv = new byte[16];
            SecureRandom secureRandom = new SecureRandom();
            secureRandom.nextBytes(newIv);
            IvParameterSpec newIvParameterSpec = new IvParameterSpec(newIv);

            Cipher newCipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            newCipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, newIvParameterSpec);

            String encodedNewIv = Base64.getEncoder().encodeToString(newIv);

            String estimatedBalance = (String) recData.get("estimatedBalance");
            byte[] estimatedBalanceByte =
                newCipher.doFinal(estimatedBalance.getBytes(StandardCharsets.UTF_8));

            List<CardTransactionDto> cardTransactionDtoList = new ArrayList<>();
            List<Map<String, Object>> transactionList =
                (List<Map<String, Object>>) recData.get("transactionList");
            for (Map<String, Object> stringObjectMap : transactionList) {
                String transactionUniqueNo = (String) stringObjectMap.get("transactionUniqueNo");
                byte[] transactionUniqueNoByte =
                    newCipher.doFinal(Base64.getDecoder().decode(transactionUniqueNo));
                String merchantId = (String) stringObjectMap.get("merchantId");
                byte[] merchantIdByte =
                    newCipher.doFinal(merchantId.getBytes(StandardCharsets.UTF_8));
                String billStatementYn = (String) stringObjectMap.get("billStatementsYn");
                byte[] billStatementYnByte =
                    newCipher.doFinal(billStatementYn.getBytes(StandardCharsets.UTF_8));
                String transactionBalance = (String) stringObjectMap.get("transactionBalance");
                byte[] transactionBalanceByte =
                    newCipher.doFinal(transactionBalance.getBytes(StandardCharsets.UTF_8));
                String transactionDate = (String) stringObjectMap.get("transactionDate");
                byte[] transactionDateByte =
                    newCipher.doFinal(transactionDate.getBytes(StandardCharsets.UTF_8));
                String transactionTime = (String) stringObjectMap.get("transactionTime");
                byte[] transactionTimeByte =
                    newCipher.doFinal(transactionTime.getBytes(StandardCharsets.UTF_8));
                String categoryName = (String) stringObjectMap.get("categoryName");
                byte[] categoryNameByte =
                    newCipher.doFinal(categoryName.getBytes(StandardCharsets.UTF_8));
                String categoryId = (String) stringObjectMap.get("categoryId");
                byte[] categoryIdByte =
                    newCipher.doFinal(categoryId.getBytes(StandardCharsets.UTF_8));
                String cardStatus = (String) stringObjectMap.get("cardStatus");
                byte[] cardStatusByte =
                    newCipher.doFinal(cardStatus.getBytes(StandardCharsets.UTF_8));
                String merchantName = (String) stringObjectMap.get("merchantName");
                byte[] merchantNameByte =
                    newCipher.doFinal(merchantName.getBytes(StandardCharsets.UTF_8));

                CardTransactionDto cardTransactionDto = CardTransactionDto.builder()
                    .transactionUniqueNo(
                        Base64.getEncoder().encodeToString(transactionUniqueNoByte))
                    .merchantId(Base64.getEncoder().encodeToString(merchantIdByte))
                    .billStatementsYn(Base64.getEncoder().encodeToString(billStatementYnByte))
                    .transactionBalance(Base64.getEncoder().encodeToString(transactionBalanceByte))
                    .transactionDate(Base64.getEncoder().encodeToString(transactionDateByte))
                    .transactionTime(Base64.getEncoder().encodeToString(transactionTimeByte))
                    .categoryName(Base64.getEncoder().encodeToString(categoryNameByte))
                    .categoryId(Base64.getEncoder().encodeToString(categoryIdByte))
                    .cardStatus(Base64.getEncoder().encodeToString(cardStatusByte))
                    .merchantName(Base64.getEncoder().encodeToString(merchantNameByte))
                    .build();

                cardTransactionDtoList.add(cardTransactionDto);
            }

            ResponseFindCardTransactionList response = ResponseFindCardTransactionList.builder()
                .iv(encodedNewIv)
                .estimatedBalance(Base64.getEncoder().encodeToString(estimatedBalanceByte))
                .transactionList(cardTransactionDtoList)
                .build();

            return response;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public ResponseOpenAccountAuth openAccountAuth(
        RequestOpenAccountAuth request,
        CustomUserDetails customUserDetails) {
        try {
            User user = userRepository.findByUserPk(customUserDetails.getUserPk())
                .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 회원"));

            String encodedKey = customUserDetails.getAesKey();
            String encodedIv = request.getIv();

            byte[] decodedKey = Base64.getDecoder().decode(encodedKey);
            byte[] decodedIv = Base64.getDecoder().decode(encodedIv);
            byte[] decodedAccountNo = Base64.getDecoder().decode(request.getAccountNo());

            SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");
            IvParameterSpec ivParameterSpec = new IvParameterSpec(decodedIv);

            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec);

            byte[] decryptedByte = cipher.doFinal(decodedAccountNo);
            String accountNo = new String(decryptedByte, StandardCharsets.UTF_8);

            RequestCreateAccountAuth requestApi = RequestCreateAccountAuth.builder()
                .accountNo(accountNo)
                .userKey(customUserDetails.getUserKey())
                .build();
            authAPI.createAccountAuth(requestApi);

            LocalDate currentDate = LocalDate.now();

            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd");
            String currentString = currentDate.format(formatter);

            RequestFindTransactionList requestTransaction = new RequestFindTransactionList();
            requestTransaction.setAccountNo(accountNo);
            requestTransaction.setUserKey(customUserDetails.getUserKey());
            requestTransaction.setOrderByType("DESC");
            requestTransaction.setTransactionType("M");
            requestTransaction.setEndDate(currentString);
            requestTransaction.setStartDate(currentString);

            Map<String, Object> apiData = demandDepositAPI.findTransactionList(requestTransaction);
            Map<String, Object> responseData = (Map<String, Object>) apiData.get("apiResponse");
            Map<String, Object> recData = (Map<String, Object>) responseData.get("REC");
            List<Map<String, Object>> transacionList = (List<Map<String, Object>>) recData.get(
                "list");
            Map<String, Object> transactionData = transacionList.get(0);
            String transactionSummary = (String) transactionData.get("transactionSummary");
            String[] authCodeSplit = transactionSummary.split(" ");

            ResponseOpenAccountAuth response = new ResponseOpenAccountAuth();

            byte[] newIv = new byte[16];
            SecureRandom secureRandom = new SecureRandom();
            secureRandom.nextBytes(newIv);
            IvParameterSpec newIvParameterSpec = new IvParameterSpec(newIv);

            Cipher newCipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            newCipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, newIvParameterSpec);

            String encodedNewIv = Base64.getEncoder().encodeToString(newIv);

            String authCode = authCodeSplit[1];

            String fcmToken = user.getFcmToken();
            String title = "1원 송금";
            String message = "[MarshMellow] 인증번호 [" + authCode + "]를 입력해주세요 사칭/전화사기에 주의하세요";
            alertService.sendNotification(fcmToken, title, message);

            byte[] authCodeByte = newCipher.doFinal(authCode.getBytes(StandardCharsets.UTF_8));

            response.setAuthCode(Base64.getEncoder().encodeToString(authCodeByte));
            response.setIv(encodedNewIv);

            return response;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public ResponseCheckAccountAuth checkAccountAuth(
        RequestCheckAccountAuth request,
        CustomUserDetails customUserDetails) {
        try {
            String encodedKey = customUserDetails.getAesKey();
            String encodedIv = request.getIv();

            byte[] decodedKey = Base64.getDecoder().decode(encodedKey);
            byte[] decodedIv = Base64.getDecoder().decode(encodedIv);
            byte[] decodedAccountNo = Base64.getDecoder().decode(request.getAccountNo());
            byte[] decodedAuthCode = Base64.getDecoder().decode(request.getAuthCode());

            SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");
            IvParameterSpec ivParameterSpec = new IvParameterSpec(decodedIv);

            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec);

            byte[] decryptedByte = cipher.doFinal(decodedAccountNo);
            byte[] decryptedByte2 = cipher.doFinal(decodedAuthCode);
            String accountNo = new String(decryptedByte, StandardCharsets.UTF_8);
            String authCode = new String(decryptedByte2, StandardCharsets.UTF_8);

            com.gbh.gbh_mm.finance.auth.vo.request.RequestCheckAccountAuth requestApi =
                com.gbh.gbh_mm.finance.auth.vo.request.RequestCheckAccountAuth.builder()
                    .accountNo(accountNo)
                    .authCode(authCode)
                    .userKey(customUserDetails.getUserKey())
                    .build();

            Map<String, Object> apiData = authAPI.checkAccountAuth(requestApi);
            Map<String, Object> responseData = (Map<String, Object>) apiData.get("apiResponse");
            Map<String, Object> recData = (Map<String, Object>) responseData.get("REC");
            String checkStatus = (String) recData.get("status");
            if (checkStatus.equals("SUCCESS")) {
                User user = userRepository.findByUserKey(customUserDetails.getUserKey());

                if (user.getUserKey() != null) {
                    byte[] newIv = new byte[16];
                    SecureRandom secureRandom = new SecureRandom();
                    secureRandom.nextBytes(newIv);
                    IvParameterSpec newIvParameterSpec = new IvParameterSpec(newIv);

                    Cipher newCipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
                    newCipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, newIvParameterSpec);

                    WithdrawalAccount withdrawalAccount = WithdrawalAccount.builder()
                        .accountNo(accountNo)
                        .user(user)
                        .build();

                    WithdrawalAccount savedWithdrawalAccount = withdrawalAccountRepository.save(
                        withdrawalAccount);

                    ResponseCheckAccountAuth response = ResponseCheckAccountAuth.builder()
                        .status("SUCCESS")
                        .withdrawalAccountId(savedWithdrawalAccount.getWithdrawalAccountId())
                        .build();

                    return response;
                } else {
                    ResponseCheckAccountAuth response = ResponseCheckAccountAuth.builder()
                        .status("FAIL")
                        .build();

                    return response;
                }
            } else {
                return null;
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public ResponseFindWithdrawalAccountList findWithdrawalAccountList
        (CustomUserDetails customUserDetails) {
        List<WithdrawalAccount> withdrawalAccountList = withdrawalAccountRepository
            .findByUser_UserPk(customUserDetails.getUserPk());

        try {
            String encodedKey = customUserDetails.getAesKey();

            byte[] decodedKey = Base64.getDecoder().decode(encodedKey);


            SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");

            byte[] newIv = new byte[16];
            SecureRandom secureRandom = new SecureRandom();
            secureRandom.nextBytes(newIv);
            IvParameterSpec newIvParameterSpec = new IvParameterSpec(newIv);

            Cipher newCipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            newCipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, newIvParameterSpec);

            String encodedNewIv = Base64.getEncoder().encodeToString(newIv);

            List<WithdrawalAccountDto> withdrawalAccountDtoList = new ArrayList<>();
            for (WithdrawalAccount withdrawalAccount : withdrawalAccountList) {
                byte[] accountByte =
                    newCipher.doFinal(withdrawalAccount.getAccountNo().getBytes(StandardCharsets.UTF_8));

                WithdrawalAccountDto withdrawalAccountDto = new WithdrawalAccountDto();
                withdrawalAccountDto.setWithdrawalAccountId(withdrawalAccount.getWithdrawalAccountId());
                withdrawalAccountDto.setAccountNo(Base64.getEncoder().encodeToString(accountByte));

                withdrawalAccountDtoList.add(withdrawalAccountDto);
            }

            ResponseFindWithdrawalAccountList response = ResponseFindWithdrawalAccountList.builder()
                .iv(encodedNewIv)
                .withdrawalAccountList(withdrawalAccountDtoList)
                .build();

            return response;

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public ResponseDeleteWithdrawalAccount deleteWithdrawalAccount(
        RequestDeleteWithdrawalAccount request) {

        try {
            Optional<WithdrawalAccount> withdrawalAccount =
                withdrawalAccountRepository.findById(request.getWithdrawalAccountId());
            if (withdrawalAccount.isPresent()) {
                withdrawalAccountRepository.delete(withdrawalAccount.get());

                ResponseDeleteWithdrawalAccount response = ResponseDeleteWithdrawalAccount.builder()
                    .message("삭제 성공")
                    .build();

                return response;
            }

        } catch (Exception e) {
            new EntityNotFoundException("존재하지 않는 출금계좌");
        }
        return null;
    }

    @Override
    public ResponseAccountTransfer accountTransfer(
        RequestWithdrawalAccountTransfer request,
        CustomUserDetails customUserDetails
    ) {
        try {
            WithdrawalAccount withdrawalAccount = withdrawalAccountRepository
                .findById(request.getWithdrawalAccountId())
                .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 출금계죄"));

            String encodedKey = customUserDetails.getAesKey();
            String encodedIv = request.getIv();

            byte[] decodedKey = Base64.getDecoder().decode(encodedKey);
            byte[] decodedIv = Base64.getDecoder().decode(encodedIv);
            byte[] decodedAccountNo = Base64.getDecoder().decode(request.getDepositAccountNo());
            byte[] decodedSummary = Base64.getDecoder().decode(request.getTransactionSummary());
            byte[] decodedBalance = Base64.getDecoder().decode(request.getTransactionBalance());

            SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");
            IvParameterSpec ivParameterSpec = new IvParameterSpec(decodedIv);

            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec);

            byte[] decryptedAccountNo = cipher.doFinal(decodedAccountNo);
            byte[] decryptedSummary = cipher.doFinal(decodedSummary);
            byte[] decryptedTransactionBalance = cipher.doFinal(decodedBalance);
            String depositAccountNo = new String(decryptedAccountNo, StandardCharsets.UTF_8);
            String summary = new String(decryptedSummary, StandardCharsets.UTF_8);
            String transactionBalanceStr =
                new String(decryptedTransactionBalance, StandardCharsets.UTF_8);
            long transactionBalance = Long.parseLong(transactionBalanceStr);

            String userKey = withdrawalAccount.getUser().getUserKey();

            RequestAccountTransfer apiRequest = new RequestAccountTransfer();
            apiRequest.setUserKey(userKey);
            apiRequest.setDepositAccountNo(depositAccountNo);
            apiRequest.setDepositTransactionSummary
                ("MarshMellow 입금 " + withdrawalAccount.getUser().getUserName());
            apiRequest.setWithdrawalAccountNo(withdrawalAccount.getAccountNo());
            apiRequest.setWithdrawalTransactionSummary(summary);
            apiRequest.setTransactionBalance(transactionBalance);

            demandDepositAPI.accountTransfer(apiRequest);

            ResponseAccountTransfer response = ResponseAccountTransfer.builder()
                .message("이체 성공")
                .build();

            return response;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public ResponseAuthTest authTest() {
        try {
            User user = userRepository.findByUserPk(3L).orElseThrow();
            String encodedKey = user.getAesKey();
            byte[] decodedKey = Base64.getDecoder().decode(encodedKey);
            SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");

            ResponseAuthTest response = new ResponseAuthTest();
            response.setEncodeKey(encodedKey);

            byte[] iv = new byte[16];
            SecureRandom secureRandom = new SecureRandom();
            secureRandom.nextBytes(iv);
            IvParameterSpec ivParameterSpec = new IvParameterSpec(iv);

            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.ENCRYPT_MODE, secretKeySpec, ivParameterSpec);

            String encodedIV = Base64.getEncoder().encodeToString(iv);

            response.setIv(encodedIV);

            String planeText = "1003857501037738";
            String planeText2 = "143";
            byte[] planeBytes = cipher.doFinal(planeText.getBytes(StandardCharsets.UTF_8));
            byte[] planeBytes2 = cipher.doFinal(planeText2.getBytes(StandardCharsets.UTF_8));

            String cipherText = Base64.getEncoder().encodeToString(planeBytes);
            String cipherText2 = Base64.getEncoder().encodeToString(planeBytes2);

            response.setValue(cipherText);
            response.setValue2(cipherText2);

            return response;

        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        } catch (NoSuchPaddingException e) {
            throw new RuntimeException(e);
        } catch (InvalidAlgorithmParameterException e) {
            throw new RuntimeException(e);
        } catch (InvalidKeyException e) {
            throw new RuntimeException(e);
        } catch (IllegalBlockSizeException e) {
            throw new RuntimeException(e);
        } catch (BadPaddingException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public ResponseAuthTest decodeTest(RequestDecodeTest request) {
        String encodedIV = request.getIv();
        String encodedCipherText = request.getValue();

        // SecretKey는 암호화할 때 사용했던 키 (예: Base64로 인코딩된 키를 디코딩 후 SecretKeySpec 생성)
        String base64Key = request.getKey();
        byte[] decodedKey = Base64.getDecoder().decode(base64Key);
        SecretKeySpec secretKeySpec = new SecretKeySpec(decodedKey, "AES");

        // 1. Base64 디코딩: IV와 암호문
        byte[] decodedIV = Base64.getDecoder().decode(encodedIV);
        byte[] decodedCipherText = Base64.getDecoder().decode(encodedCipherText);

        // 2. IV 객체 생성
        IvParameterSpec ivParameterSpec = new IvParameterSpec(decodedIV);

        try {

            // 3. Cipher 초기화 (AES/CBC/PKCS5Padding 모드 사용)
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.DECRYPT_MODE, secretKeySpec, ivParameterSpec);

            // 4. 복호화 수행
            byte[] decryptedBytes = cipher.doFinal(decodedCipherText);
            String decryptedText = new String(decryptedBytes, StandardCharsets.UTF_8);

            ResponseAuthTest response = new ResponseAuthTest();
            response.setValue(decryptedText);

            return response;
        } catch (NoSuchPaddingException e) {
            throw new RuntimeException(e);
        } catch (IllegalBlockSizeException e) {
            throw new RuntimeException(e);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        } catch (InvalidAlgorithmParameterException e) {
            throw new RuntimeException(e);
        } catch (BadPaddingException e) {
            throw new RuntimeException(e);
        } catch (InvalidKeyException e) {
            throw new RuntimeException(e);
        }
    }
}
