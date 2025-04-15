package com.gbh.gbh_mm.asset.controller;

import com.gbh.gbh_mm.asset.RequestDecodeTest;
import com.gbh.gbh_mm.asset.ResponseAuthTest;
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
import com.gbh.gbh_mm.asset.service.AssetService;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindAssetList;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCreateAccountAuth;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindTransactionList;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindPayment;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFindRepaymentList;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsPayment;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/asset")
@AllArgsConstructor
public class AssetController {

    private final AssetService assetService;

    /* 자산 목록 조회 */
    @GetMapping
    public ResponseEntity<ResponseFindAssetList> findAssetList(
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseFindAssetList response = assetService.findAssetList(customUserDetails);

        return ResponseEntity.ok(response);
    }

    /* 입출금 계좌 내역 조회 */
    @GetMapping("/deposit-demand-transaction")
    public ResponseEntity<ResponseFindDepositDemandTransactionList> findDepositDemandTransactionList(
        @RequestBody RequestFindDepositDemandTransactionList request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseFindDepositDemandTransactionList response = assetService
            .findDepositDemandTransactionList(request, customUserDetails);

        return ResponseEntity.ok(response);
    }

    /* 예금 납입 내역 조회 */
    @GetMapping("/deposit-payment")
    public ResponseEntity<ResponseFindDepositPayment> findDepositPayment(
        @RequestBody RequestFindDepositPayment request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseFindDepositPayment response = assetService
            .findDepositPayment(request, customUserDetails);

        return ResponseEntity.ok(response);
    }

    /* 적금 계좌 내역 조회 */
    @GetMapping("/savings-payment")
    public ResponseEntity<ResponseFindSavingsPaymentList> findSavingsPaymentList(
        @RequestBody RequestFindSavingsPaymentList request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseFindSavingsPaymentList response = assetService
            .findSavingsPaymentList(request, customUserDetails);

        return ResponseEntity.ok(response);
    }

    /* 대출 상환 내역 조회 */
    @GetMapping("/loan-payment")
    public ResponseEntity<ResponseFindLoanPaymentList> findLoanPaymentList(
        @RequestBody RequestFindLoanPaymentList request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseFindLoanPaymentList response = assetService
            .findLoanPaymentList(request, customUserDetails);

        return ResponseEntity.ok(response);
    }

    /* 카드 내역 조회 */
    @GetMapping("/card-transaction")
    public ResponseEntity<ResponseFindCardTransactionList> findCardTransactionList(
        @RequestBody RequestFindCardTransactionList request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseFindCardTransactionList response = assetService
            .findCardTransactionList(request, customUserDetails);

        return ResponseEntity.ok(response);
    }

    /* 1원 송금 */
    @PostMapping("/open-account-auth")
    public ResponseEntity<ResponseOpenAccountAuth> openAccountAuth(
        @RequestBody RequestOpenAccountAuth request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseOpenAccountAuth response = assetService
            .openAccountAuth(request, customUserDetails);

        return ResponseEntity.ok(response);
    }

    /* 1원 송금 인증 */
    @PostMapping("/check-account-auth")
    public ResponseEntity<ResponseCheckAccountAuth> checkAccountAuth(
        @RequestBody RequestCheckAccountAuth request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseCheckAccountAuth response = assetService
            .checkAccountAuth(request, customUserDetails);

        return ResponseEntity.ok(response);
    }

    /* 출금 계좌 목록 */
    @GetMapping("/withdrawal-account")
    public ResponseEntity<ResponseFindWithdrawalAccountList> findWithdrawalAccountList(
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseFindWithdrawalAccountList response = assetService
            .findWithdrawalAccountList(customUserDetails);

        return ResponseEntity.ok(response);
    }

    /* 출금 계좌 삭제 */
    @DeleteMapping("/withdrawal-account")
    public ResponseEntity<ResponseDeleteWithdrawalAccount> deleteWithdrawalAccount(
        @RequestBody RequestDeleteWithdrawalAccount request
    ) {
        ResponseDeleteWithdrawalAccount response = assetService.deleteWithdrawalAccount(request);

        return ResponseEntity.ok(response);
    }

    /* 계좌 송금 */
    @PostMapping("/account-transfer")
    public ResponseEntity<ResponseAccountTransfer> accountTransfer(
        @RequestBody RequestWithdrawalAccountTransfer request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseAccountTransfer response = assetService.accountTransfer(request, customUserDetails);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/auth-test")
    public ResponseAuthTest authTest() {
        return assetService.authTest();
    }

    @GetMapping("/decode-test")
    public ResponseAuthTest decodeTest(
        @RequestBody RequestDecodeTest request
    ) {
        return assetService.decodeTest(request);
    }

}
