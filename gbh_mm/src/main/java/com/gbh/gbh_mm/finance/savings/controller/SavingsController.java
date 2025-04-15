package com.gbh.gbh_mm.finance.savings.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestCreateAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestDeleteAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestEarlyInterest;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccountList;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindInterest;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindPayment;
import com.gbh.gbh_mm.finance.savings.service.SavingsService;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestCreateSavingsAccount;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestCreateSavingsProduct;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestDeleteSavingsAccount;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsAccount;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsAccountList;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsInterest;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestFindSavingsPayment;
import com.gbh.gbh_mm.finance.savings.vo.request.RequestSavingsEalryInterest;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/finance/savings")
@AllArgsConstructor
public class SavingsController {
    private final SavingsService savingsService;

    @PostMapping("/product")
    public ResponseEntity<Map<String, Object>> createProduct(
        @RequestBody RequestCreateSavingsProduct request
    ) throws JsonProcessingException {
        Map<String, Object> response = savingsService.createProduct(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/product-list")
    public ResponseEntity<Map<String, Object>> findProductList() throws JsonProcessingException {
        Map<String, Object> response = savingsService.findProductList();

        return ResponseEntity.ok(response);
    }

    @PostMapping("/account")
    public ResponseEntity<Map<String, Object>> createAccount(
        @RequestBody RequestCreateSavingsAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = savingsService.createAccount(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/account-list")
    public ResponseEntity<Map<String, Object>> findAccountList(
        @RequestBody RequestFindSavingsAccountList request
    ) throws JsonProcessingException {
        Map<String, Object> response = savingsService.findAccountList(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/account")
    public ResponseEntity<Map<String, Object>> findAccount(
        @RequestBody RequestFindSavingsAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = savingsService.findAccount(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/payment")
    public ResponseEntity<Map<String, Object>> findPayment(
        @RequestBody RequestFindSavingsPayment request
    ) throws JsonProcessingException {
        Map<String, Object> response = savingsService.findPayment(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/interest")
    public ResponseEntity<Map<String, Object>> findInterest(
        @RequestBody RequestFindSavingsInterest request
    ) throws JsonProcessingException {
        Map<String, Object> response = savingsService.findInterest(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/early-interest")
    public ResponseEntity<Map<String, Object>> findEarlyInterest(
        @RequestBody RequestSavingsEalryInterest request
    ) throws JsonProcessingException {
        Map<String, Object> response = savingsService.findEarlyInterest(request);

        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/account")
    public ResponseEntity<Map<String, Object>> deleteAccount(
        @RequestBody RequestDeleteSavingsAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = savingsService.deleteAccount(request);

        return ResponseEntity.ok(response);
    }
}
