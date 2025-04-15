package com.gbh.gbh_mm.finance.auth.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.auth.service.AuthService;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCheckAccountAuth;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCreateAccountAuth;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/finance/auth")
@AllArgsConstructor
public class AuthController {
    private final AuthService authService;

    @PostMapping("/open-account-auth")
    public ResponseEntity<Map<String, Object>> createAccountAuth(
            @RequestBody RequestCreateAccountAuth request
    ) throws JsonProcessingException {
        Map<String, Object> response = authService.createAccountAuth(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/check-account-auth")
    public ResponseEntity<Map<String, Object>> checkAccountAuth(
            @RequestBody RequestCheckAccountAuth request
    ) throws JsonProcessingException {
        Map<String, Object> response = authService.checkAccountAuth(request);

        return ResponseEntity.ok(response);
    }
}
