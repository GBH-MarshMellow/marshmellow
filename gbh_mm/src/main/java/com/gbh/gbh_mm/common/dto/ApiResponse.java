package com.gbh.gbh_mm.common.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class ApiResponse<T> {

    private int code;
    private String message;
    private T data;
}
