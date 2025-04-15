package com.gbh.gbh_mm.finance.user.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.finance.user.service.UserFinService;
import com.gbh.gbh_mm.finance.user.vo.request.RequestCreateUserKey;
import com.gbh.gbh_mm.finance.user.vo.request.RequestReissueUserKey;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/finance/user")
@AllArgsConstructor
public class UserFinController {
    private final UserFinService userFinService;

    @PostMapping("/user-key")
    private ResponseEntity<Map<String, Object>> createUserKey(
        @RequestBody RequestCreateUserKey request
    ) throws JsonProcessingException {
        Map<String, Object> response = userFinService.createUserKey(request);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/search")
    private ResponseEntity<Map<String, Object>> searchUser(
        @RequestBody RequestReissueUserKey request
    ) throws JsonProcessingException {
        Map<String, Object> response = userFinService.searchUser(request);

        return ResponseEntity.ok(response);
    }
}
