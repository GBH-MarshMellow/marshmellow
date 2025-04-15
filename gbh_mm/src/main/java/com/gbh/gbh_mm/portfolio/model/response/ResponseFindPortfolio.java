package com.gbh.gbh_mm.portfolio.model.response;

import com.gbh.gbh_mm.portfolio.model.dto.PortfolioCategoryDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class ResponseFindPortfolio {
    private int portfolioPk;
    private String fileUrl;
    private String createDate;
    private String createTime;
    private String originFileName;
    private String fileName;
    private String portfolioMemo;
    private PortfolioCategoryDto portfolioCategory;
}
