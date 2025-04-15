package com.gbh.gbh_mm.finance.card.vo.request;

import java.util.List;
import lombok.Data;

@Data
public class RequestCreateCardProduct {
    private String cardIssuerCode;
    private String cardName;
    private long baselinePerformance;
    private long maxBenefitLimit;
    private String cardDescription;
    private List<CardBenefits> cardBenefits;

}
