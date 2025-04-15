package com.gbh.gbh_mm.user.model.request;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class LoginByPinRequestDto {

    private String phoneNumber;

    private String pin;

}
