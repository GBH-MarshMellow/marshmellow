package com.gbh.gbh_mm.finance.card.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.card.service.CardService;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateCardProduct;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateMerchant;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateTransaction;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateUserCard;
import com.gbh.gbh_mm.finance.card.vo.request.RequestDeleteTransaction;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindBilling;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindCardTransactionList;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindUserCardList;
import com.gbh.gbh_mm.finance.card.vo.request.RequestUpdateAccount;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindTransactionList;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/finance/card")
@AllArgsConstructor
public class CardController {

    private final CardService cardService;

    @GetMapping("/category")
    public ResponseEntity<Map<String, Object>> findCategoryList() throws JsonProcessingException {
        Map<String, Object> response = cardService.findCategoryList();

        return ResponseEntity.ok(response);
    }

    @PostMapping("/merchant")
    public ResponseEntity<Map<String, Object>> createMerchant(
        @RequestBody RequestCreateMerchant request) throws JsonProcessingException {
        Map<String, Object> response = cardService.createMerchant(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/merchant-list")
    public ResponseEntity<Map<String, Object>> findMerchantList() throws JsonProcessingException {
        Map<String, Object> response = cardService.findMerchantList();

        return ResponseEntity.ok(response);
    }

    @GetMapping("/company")
    public ResponseEntity<Map<String, Object>> findCompanyList() throws JsonProcessingException {
        Map<String, Object> response = cardService.findCompanyList();

        return ResponseEntity.ok(response);
    }

    @PostMapping("/product")
    public ResponseEntity<Map<String, Object>> createProduct(
        @RequestBody RequestCreateCardProduct request
    ) throws JsonProcessingException {
        Map<String, Object> response = cardService.createProduct(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/product-list")
    public ResponseEntity<Map<String, Object>> findProductList() throws JsonProcessingException {
        Map<String, Object> response = cardService.findProductList();

        return ResponseEntity.ok(response);
    }

    @PostMapping("/user-card")
    public ResponseEntity<Map<String, Object>> createUserCard(
        @RequestBody RequestCreateUserCard request
    ) throws JsonProcessingException {
        Map<String, Object> response = cardService.createUserCard(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/user-card-list")
    public ResponseEntity<Map<String, Object>> findUserCardList(
        @RequestBody RequestFindUserCardList request
    ) throws JsonProcessingException {
        Map<String, Object> response = cardService.findUserCardList(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/transaction")
    private ResponseEntity<Map<String, Object>> createTransaction(
        @RequestBody RequestCreateTransaction request
    ) throws JsonProcessingException {
        Map<String, Object> response = cardService.createTransaction(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/transaction-list")
    private ResponseEntity<Map<String, Object>> findTransactionList(
        @RequestBody RequestFindCardTransactionList request
    ) throws JsonProcessingException {
        Map<String, Object> response = cardService.findTransactionList(request);

        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/transaction")
    public ResponseEntity<Map<String, Object>> deleteTransaction(
        @RequestBody RequestDeleteTransaction request
    ) throws JsonProcessingException {
        Map<String, Object> response = cardService.deleteTransction(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/billing")
    public ResponseEntity<Map<String, Object>> findBilling(
        @RequestBody RequestFindBilling request
    ) throws JsonProcessingException {
        Map<String, Object> response = cardService.findBilling(request);

        return ResponseEntity.ok(response);
    }

    @PutMapping("/account")
    public ResponseEntity<Map<String, Object>> updateAccount(
        @RequestBody RequestUpdateAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = cardService.updateAccount(request);

        return ResponseEntity.ok(response);
    }
}
