package com.gbh.gbh_mm.common.dto;

import com.gbh.gbh_mm.common.exception.ErrorResponseDto;
import org.springframework.core.MethodParameter;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.servlet.mvc.method.annotation.ResponseBodyAdvice;

import java.util.List;

@RestControllerAdvice
public class CommonControllerAdvice implements ResponseBodyAdvice<Object> {
    /**
     * returnType이 void가 아닌 경우만 감싸도록 설정 (선택 사항)
     */
    @Override
    public boolean supports(MethodParameter returnType, Class<? extends
            HttpMessageConverter<?>> converterType) {
        return MappingJackson2HttpMessageConverter.class.isAssignableFrom(converterType);
    }
    /**
     * 응답 바디가 작성되기 전(body -> HttpMessageConverter로 변환되기 전)에
     * 공통 응답 포맷으로 감싸는 로직을 처리
     */
    @Override
    public Object beforeBodyWrite(Object body, MethodParameter returnType, MediaType selectedContentType, Class<? extends HttpMessageConverter<?>> selectedConverterType, ServerHttpRequest request, ServerHttpResponse response) {
        if (body instanceof ApiResponse) {
            return body;
        }
        // 에러일 경우: ErrorResponse 혹은 List<ErrorResponse> 라고 가정
        // 1️⃣ 단건 ErrorResponseDto 처리
        if (body instanceof ErrorResponseDto errorResponse) {
            return wrapResponse(errorResponse.getStatus(), errorResponse.getMessage());
        }

        // 2️⃣ List<ErrorResponseDto> 처리
        if (body instanceof List<?> list && !list.isEmpty() && list.getFirst() instanceof ErrorResponseDto firstError) {
            return wrapResponse(firstError.getStatus(), firstError.getMessage());
        }
        return wrapResponse(body, HttpStatus.OK.value());
    }

    private ApiResponse<Object> wrapResponse(Object data, int code) {
        return ApiResponse.builder()
                .code(code)
                .message("성공")
                .data(data)
                .build();
    }
    private ApiResponse<Object> wrapResponse(int code, String message) {
        return ApiResponse.builder()
                .code(code)
                .message(message)
                .build();
    }
}
