package com.gbh.gbh_mm.finance.deposit.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.deposit.service.DepositService;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestCreateAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestCreateDepositProduct;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestDeleteAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestEarlyInterest;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccount;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindAccountList;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindInterest;
import com.gbh.gbh_mm.finance.deposit.vo.request.RequestFindPayment;
import java.util.HashMap;
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
@RequestMapping("/finance/deposit")
@AllArgsConstructor
public class DepositController {
    private final DepositService depositService;

    @PostMapping("/product")
    public ResponseEntity<Map<String, Object>> createProduct(
        @RequestBody RequestCreateDepositProduct request) throws JsonProcessingException {

        Map<String, Object> response = depositService.createDepositProduct(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/product-list")
    public ResponseEntity<Map<String, Object>> findProductList() throws JsonProcessingException {
        Map<String, Object> response = depositService.findProductList();

        return ResponseEntity.ok(response);
    }

    @PostMapping("/account")
    public ResponseEntity<Map<String, Object>> createAccount(
        @RequestBody RequestCreateAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = depositService.createAccount(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/account-list")
    public ResponseEntity<Map<String, Object>> findAccountList(
        @RequestBody RequestFindAccountList request
    ) throws JsonProcessingException {
        Map<String, Object> response = depositService.findAccountList(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/account")
    public ResponseEntity<Map<String, Object>> findAccount(
        @RequestBody RequestFindAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = depositService.findAccount(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/payment")
    public ResponseEntity<Map<String, Object>> findPayment(
        @RequestBody RequestFindPayment request
    ) throws JsonProcessingException {
        Map<String, Object> response = depositService.findPayment(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/interest")
    public ResponseEntity<Map<String, Object>> findInterest(
        @RequestBody RequestFindInterest request
    ) throws JsonProcessingException {
        Map<String, Object> response = depositService.findInterest(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/early-interest")
    public ResponseEntity<Map<String, Object>> findEarlyInterest(
        @RequestBody RequestEarlyInterest request
    ) throws JsonProcessingException {
        Map<String, Object> response = depositService.findEarlyInterest(request);

        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/account")
    public ResponseEntity<Map<String, Object>> deleteAccount(
        @RequestBody RequestDeleteAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = depositService.deleteAccount(request);

        return ResponseEntity.ok(response);
    }
}
