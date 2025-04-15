package com.gbh.gbh_mm.finance.demandDeposit.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.demandDeposit.service.DemandDepositFinService;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestAccountTransfer;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestCreateDemandDepositAccount;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestCreateDepositProduct;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestDeleteDemandDepositAccount;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestDemandDepositDeposit;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestDemandDepositWithdrawal;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindBalance;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindDemandDepositAccount;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindDemandDepositAccountList;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindHolderName;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindTransactionList;
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
@RequestMapping("/finance/demand-deposit")
@AllArgsConstructor
public class DemandDepositFinController {

    private final DemandDepositFinService demandDepositFinService;

    @GetMapping("/product")
    private ResponseEntity<Map<String, Object>> findDepositList() throws JsonProcessingException {
        Map<String, Object> response = demandDepositFinService.findDepositList();

        return ResponseEntity.ok(response);
    }

    @PostMapping("/product")
    private ResponseEntity<Map<String, Object>> createDepositProduct(
        @RequestBody RequestCreateDepositProduct request
    ) throws JsonProcessingException {
        Map<String, Object> response = demandDepositFinService.createDepositProduct(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/account")
    private ResponseEntity<Map<String, Object>> createDemandDepositAccount(
        @RequestBody RequestCreateDemandDepositAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = demandDepositFinService.createDemandDepositAccount(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/account-list")
    private ResponseEntity<Map<String, Object>> findDemandDepositAccountList(
        @RequestBody RequestFindDemandDepositAccountList request
    )
        throws JsonProcessingException {
        Map<String, Object> response = demandDepositFinService.findDemandDespositAccountList(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/account")
    private ResponseEntity<Map<String, Object>> findDemandDepositAccount(
        @RequestBody RequestFindDemandDepositAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = demandDepositFinService.findDemandDespositAccount(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/holder-name")
    private ResponseEntity<Map<String, Object>> findHolderName(
        @RequestBody RequestFindHolderName request
    ) throws JsonProcessingException {
        Map<String, Object> response = demandDepositFinService.findHolderName(request);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/balance")
    private ResponseEntity<Map<String, Object>> findBalance(
        @RequestBody RequestFindBalance request
    ) throws JsonProcessingException {
        Map<String, Object> response = demandDepositFinService.findBalance(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/withdrawal")
    private ResponseEntity<Map<String, Object>> withdrawal(
        @RequestBody RequestDemandDepositWithdrawal request
    ) throws JsonProcessingException {
        Map<String, Object> response = demandDepositFinService.withdrawal(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/deposit")
    private ResponseEntity<Map<String, Object>> deposit(
        @RequestBody RequestDemandDepositDeposit request
    ) throws JsonProcessingException {
        Map<String, Object> response = demandDepositFinService.deposit(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/account-transfer")
    private ResponseEntity<Map<String, Object>> accountTransfer(
        @RequestBody RequestAccountTransfer request
    ) throws JsonProcessingException {
        Map<String, Object> reponse = demandDepositFinService.accountTransfer(request);

        return ResponseEntity.ok(reponse);
    }

    @GetMapping("/transaction-list")
    private ResponseEntity<Map<String, Object>> findTransacitonList(
        @RequestBody RequestFindTransactionList request
    ) throws JsonProcessingException {
        Map<String, Object> reponse = demandDepositFinService.findTransactionList(request);

        return ResponseEntity.ok(reponse);
    }

    @DeleteMapping("/delete")
    private ResponseEntity<Map<String, Object>> delete(
        @RequestBody RequestDeleteDemandDepositAccount request
    ) throws JsonProcessingException {
        Map<String, Object> response = demandDepositFinService.deleteDemandDepositAccount(request);

        return ResponseEntity.ok(response);
    }


}
