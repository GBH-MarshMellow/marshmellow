package com.gbh.gbh_mm.autoTransaction.controller;

import com.gbh.gbh_mm.autoTransaction.model.vo.request.RequestCreateAutoTransaction;
import com.gbh.gbh_mm.autoTransaction.model.vo.response.ResponseCreateAutoTransaction;
import com.gbh.gbh_mm.autoTransaction.model.vo.response.ResponseDemandDepositList;
import com.gbh.gbh_mm.autoTransaction.service.AutoTransactionService;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import lombok.AllArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/auto-transaction")
@AllArgsConstructor
public class AutoTransactionController {
    private final AutoTransactionService autoTransactionService;

    @PostMapping
    public ResponseCreateAutoTransaction autoTransaction(
        @RequestBody RequestCreateAutoTransaction request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseCreateAutoTransaction response =
            autoTransactionService.createAutoTransaction(request, customUserDetails);

        return response;
    }

    @GetMapping("/demand-deposit-list")
    public ResponseDemandDepositList findDemandDepostList(
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        return autoTransactionService.findDemandDepositList(customUserDetails);
    }
}
