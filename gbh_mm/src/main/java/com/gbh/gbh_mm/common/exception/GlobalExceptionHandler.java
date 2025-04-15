package com.gbh.gbh_mm.common.exception;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.HttpStatus;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.util.ArrayList;
import java.util.List;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(CustomException.class)
    public List<ErrorResponseDto> customExceptionHandler(CustomException e) {
        return List.of(ErrorResponseDto.builder()
                .status(e.getErrorCode().getStatus().value())
                .errorCode(e.getErrorCode().getCode())
                .message(e.getErrorCode().getMessage())
                .build());
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public List<ErrorResponseDto> handleValidationExceptions(MethodArgumentNotValidException ex) {
        BindingResult bindingResult = ex.getBindingResult();
        List<ErrorResponseDto> errorResponses = new ArrayList<>();

        for (FieldError fieldError : bindingResult.getFieldErrors()) {
            String errorMessage = "Field '" + fieldError.getField() + "' " + fieldError.getDefaultMessage();

            ErrorResponseDto errorResponseDto = ErrorResponseDto.builder()
                    .status(HttpStatus.BAD_REQUEST.value())
                    .errorCode("VALIDATION_ERROR")
                    .message(errorMessage)
                    .build();

            errorResponses.add(errorResponseDto);
        }
        return errorResponses;
    }

    @ExceptionHandler(WebClientResponseException.class)
    public List<ErrorResponseDto> handleWebClientExceptions(WebClientResponseException ex) {
        ObjectMapper objectMapper = new ObjectMapper();
        List<ErrorResponseDto> errorResponses = new ArrayList<>();

        try {
            JsonNode root = objectMapper.readTree(ex.getResponseBodyAsString());
            if (root.isArray()) {
                for (JsonNode node : root) {
                    errorResponses.add(ErrorResponseDto.builder()
                            .status(node.get("status").asInt())
                            .errorCode(node.get("errorCode").asText())
                            .message(node.get("message").asText())
                            .build());
                }
            }
        } catch (JsonProcessingException e) {
            errorResponses.add(ErrorResponseDto.builder()
                    .status(HttpStatus.INTERNAL_SERVER_ERROR.value())
                    .errorCode("PARSING_ERROR")
                    .message("cert 응답 파싱 실패")
                    .build());
        }

        return errorResponses;
    }
}
