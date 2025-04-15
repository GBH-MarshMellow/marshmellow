package com.gbh.gbh_mm.finance.loan.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccountList;
import com.gbh.gbh_mm.finance.loan.service.LoanService;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestCreateAudit;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestCreateLoanAccount;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestCreateLoanProduct;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFindAuditList;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFindRepaymentList;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestFullRepayment;
import com.gbh.gbh_mm.finance.loan.vo.request.RequestUserRating;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/finance/loan")
@AllArgsConstructor
public class LoanController {
    private final LoanService loanService;

    @GetMapping("/rating")
    public ResponseEntity<Map<String, Object>> findAssetRating() throws JsonProcessingException {
        Map<String, Object> response = loanService.findAssetRating();

        return ResponseEntity.ok(response);
    }

    @GetMapping("/user-rating")
    public ResponseEntity<Map<String, Object>> findUserRating(
        @RequestBody RequestUserRating request
    ) throws JsonProcessingException {
        Map<String, Object> response = loanService.findUserRating(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/product")
    public ResponseEntity<Map<String, Object>> createProduct(
        @RequestBody RequestCreateLoanProduct request
    ) throws JsonProcessingException {
        Map<String, Object> response = loanService.createProduct(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/product")
    public ResponseEntity<Map<String, Object>> findProduct() throws JsonProcessingException {
        Map<String, Object> response = loanService.findProductList();

        return ResponseEntity.ok(response);
    }

    @PostMapping("/audit")
    public ResponseEntity<Map<String, Object>> createAudit(
        @RequestBody RequestCreateAudit request
    ) throws JsonProcessingException {
        Map<String, Object> response = loanService.createAudit(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/audit-list")
    public ResponseEntity<Map<String, Object>> findAuditList(
        @RequestBody RequestFindAuditList request
    ) throws JsonProcessingException {
        Map<String, Object> response = loanService.findAuditList(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/account")
    public ResponseEntity<Map<String, Object>> createAccount(
        @RequestBody RequestCreateLoanAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = loanService.createAccount(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/account-list")
    public ResponseEntity<Map<String, Object>> findAccountList(
        @RequestBody RequestFindAccountList request
    ) throws JsonProcessingException {
        Map<String, Object> response = loanService.findAccountList(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/repayment-list")
    public ResponseEntity<Map<String, Object>> findRepaymentList(
        @RequestBody RequestFindRepaymentList request
    )
    throws JsonProcessingException {
        Map<String, Object> response = loanService.findRepaymentList(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/full-repayment")
    public ResponseEntity<Map<String, Object>> createFullRepayment(
        @RequestBody RequestFullRepayment request
    ) throws JsonProcessingException {
        Map<String, Object> response = loanService.fullRepayement(request);

        return ResponseEntity.ok(response);
    }


}
