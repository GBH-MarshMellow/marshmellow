package com.gbh.gbh_cert.model.dto.request;

import lombok.*;

@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class CertIssueRequestDto {
    private String csrPem;
    private String userEmail;
    private String connectionInformation;
    private String userName;
    private String phoneNumber;
    private String birth;
}
