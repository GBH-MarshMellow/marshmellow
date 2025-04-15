package com.gbh.gbh_cert.exception;

import org.springframework.http.HttpStatus;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

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
}
