package com.gbh.gbh_mm.user.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@AllArgsConstructor
@Builder
@Getter
public class SignUpResponseDto {
    private String accessToken;
    private String refreshToken;
}
