package com.gbh.gbh_mm.user.model.response;


import com.gbh.gbh_mm.user.model.entity.User;
import lombok.*;

@NoArgsConstructor
@AllArgsConstructor
@Builder
@Getter
public class IdentityVerificationResponseDto {

    private String serverEmail;
    private String code;
    private int expiresIn;
    private boolean verified;

    public void setVerified(boolean verified) {
        this.verified = verified;
    }
}
