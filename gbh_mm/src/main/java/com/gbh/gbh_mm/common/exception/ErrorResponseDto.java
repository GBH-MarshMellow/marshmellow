package com.gbh.gbh_mm.common.exception;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class ErrorResponseDto {

    private int status;
    private String errorCode;
    private String message;

}
