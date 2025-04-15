package com.gbh.gbh_cert.exception;

import lombok.Getter;
import org.springframework.http.HttpStatus;

@Getter
public enum ErrorCode {

    // 🔹 400번대: 클라이언트 에러
    BAD_REQUEST(HttpStatus.BAD_REQUEST, "E400", "잘못된 요청입니다."),
    VALIDATION_FAILED(HttpStatus.BAD_REQUEST, "E401", "입력값이 유효하지 않습니다."),
    JSON_PARSE_ERROR(HttpStatus.BAD_REQUEST, "E402", "요청 본문의 JSON 형식이 올바르지 않습니다."),
    UNSUPPORTED_MEDIA_TYPE(HttpStatus.UNSUPPORTED_MEDIA_TYPE, "E415", "지원하지 않는 미디어 타입입니다."),

    // 🔹 401/403: 인증 & 인가 에러
    UNAUTHORIZED(HttpStatus.UNAUTHORIZED, "E401", "인증이 필요합니다."),
    FORBIDDEN(HttpStatus.FORBIDDEN, "E403", "이 요청을 수행할 권한이 없습니다."),
    TOKEN_EXPIRED(HttpStatus.UNAUTHORIZED, "E402", "토큰이 만료되었습니다."),
    INVALID_TOKEN(HttpStatus.UNAUTHORIZED, "E403", "유효하지 않은 토큰입니다."),
    USER_INVALID_PIN(HttpStatus.UNAUTHORIZED, "E406", "유효하지 않은 PIN 입니다."),
    CERTFICATE_NOT_EQUALS(HttpStatus.BAD_REQUEST, "E411", "인증서가 서버에 등록된 것과 일치하지 않습니다"),
    // 🔹 404/409: 리소스 관련 에러
    USER_NOT_FOUND(HttpStatus.NOT_FOUND, "E404", "해당 유저를 찾을 수 없습니다."),
    CHILD_NOT_FOUND(HttpStatus.NOT_FOUND, "E404", "해당 ID를 가진 아이를 찾을 수 없습니다."),
    RESOURCE_NOT_FOUND(HttpStatus.NOT_FOUND, "E405", "요청한 리소스를 찾을 수 없습니다."),
    DUPLICATE_RESOURCE(HttpStatus.CONFLICT, "E409", "이미 존재하는 리소스입니다."),
    CONFLICT_ERROR(HttpStatus.CONFLICT, "E410", "요청이 서버 상태와 충돌합니다."),

    // 🔹 500번대: 서버 에러
    INTERNAL_SERVER_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "E500", "서버 내부 오류가 발생했습니다."),
    DATABASE_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "E501", "데이터베이스 오류가 발생했습니다."),
    FILE_UPLOAD_FAILED(HttpStatus.INTERNAL_SERVER_ERROR, "E502", "파일 업로드에 실패했습니다."),
    SERVICE_UNAVAILABLE(HttpStatus.SERVICE_UNAVAILABLE, "E503", "현재 서비스 이용이 불가능합니다."),
    TIMEOUT_ERROR(HttpStatus.GATEWAY_TIMEOUT, "E504", "서버 응답 시간이 초과되었습니다."),
    EXTERNAL_API_ERROR(HttpStatus.BAD_GATEWAY, "E505", "외부 API 호출 중 오류가 발생했습니다.");

    private final HttpStatus status;
    private final String code;  // 에러 코드 문자열 (예: "E001")
    private final String message;  // 사용자 친화적인 에러 메시지

    ErrorCode(HttpStatus status, String code, String message) {
        this.status = status;
        this.code = code;
        this.message = message;
    }
}
