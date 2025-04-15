package com.gbh.gbh_cert.model.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@AllArgsConstructor
@Builder
@Getter
public class CertResponseDto {

    String certificatePem;
    String halfUserKey;
}
