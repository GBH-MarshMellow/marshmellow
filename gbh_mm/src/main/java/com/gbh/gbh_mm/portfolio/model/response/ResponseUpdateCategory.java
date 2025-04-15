package com.gbh.gbh_mm.portfolio.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ResponseUpdateCategory {
    private int portfolioCategoryPk;
    private String portfolioCategoryName;
    private String portfolioCategoryMemo;
}
