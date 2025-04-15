package com.gbh.gbh_mm.user.model.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ClientCertIssueRequestDto {
    private String csrPem;
    private String userEmail;
}
