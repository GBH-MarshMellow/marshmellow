package com.gbh.gbh_mm.user.model.request;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class SignUpRequestDto {

    private String userName;
    private String phoneNumber;
    private String userCode;
    private String pin;
    private String fcmToken;
}
