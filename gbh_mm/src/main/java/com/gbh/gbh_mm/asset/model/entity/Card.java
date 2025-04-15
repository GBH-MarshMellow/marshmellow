package com.gbh.gbh_mm.asset.model.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Card {
    private String cardNo;
    private String cvc;
    private String cardIssuerCode;
    private String cardIssuerName;
    private String cardName;
    private String cardBalance;
}
