package com.gbh.gbh_mm.finance.bank.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.bank.service.BankFinService;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/finance/bank")
@AllArgsConstructor
public class BankFinController {

    private final BankFinService bankFinService;

    @GetMapping
    public ResponseEntity<Map<String, Object>> findBankList() throws JsonProcessingException {
        Map<String, Object> response = bankFinService.findBankList();

        return ResponseEntity.ok(response);
    }

}
