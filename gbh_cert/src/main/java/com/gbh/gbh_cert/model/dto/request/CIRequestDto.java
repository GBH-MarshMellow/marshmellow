package com.gbh.gbh_cert.model.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CIRequestDto {

    @NotBlank(message = "사용자 이름은 필수입니다.")
    @Size(min = 2, max = 10, message = "사용자 이름은 2~10자 사이여야 합니다.")
    private String userName;

    @NotBlank(message = "전화번호는 필수입니다.")
    @Pattern(regexp = "^010\\d{8}$", message = "전화번호는 010으로 시작하는 11자리 숫자여야 합니다.")
    @Size(min = 11, max = 11, message = "전화번호는 11자리여야 합니다.")
    private String phoneNumber;

    @NotBlank(message = "사용자 코드는 필수입니다.")
    @Pattern(regexp = "^\\d{6}-[1-4]\\d*", message = "유저 코드는 숫자 6자리-성별 숫자 형식이어야 합니다.")
    @Size(min = 8, max = 8, message = "유저 코드는 정확히 8자리여야 합니다.")
    private String userCode;
}
