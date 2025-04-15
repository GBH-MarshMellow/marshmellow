package com.gbh.gbh_mm.delusion.contoller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.delusion.response.AvailableAmountResponseDto;
import com.gbh.gbh_mm.delusion.response.AverageSpendingResponseDto;
import com.gbh.gbh_mm.delusion.service.DelusionService;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import lombok.AllArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/delusion")
@AllArgsConstructor
public class DelusionController {

    private DelusionService delusionService;

    @GetMapping
    public AvailableAmountResponseDto getAvailableAmount(@AuthenticationPrincipal CustomUserDetails customUserDetails){
        return delusionService.getAvailableAmount(customUserDetails);
    }

    @GetMapping("/average")
    public AverageSpendingResponseDto getAverageSpending(@AuthenticationPrincipal CustomUserDetails customUserDetails) throws JsonProcessingException {
        return delusionService.getAverageSpending(customUserDetails);
    }

}
