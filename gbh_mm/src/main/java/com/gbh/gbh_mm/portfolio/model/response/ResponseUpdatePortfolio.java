package com.gbh.gbh_mm.portfolio.model.response;

import com.gbh.gbh_mm.portfolio.model.dto.PortfolioCategoryDto;
import lombok.Data;

@Data
public class ResponseUpdatePortfolio {
    private int portfolioPk;
    private String fileUrl;
    private String createDate;
    private String createTime;
    private String originFileName;
    private String fileName;
    private String portfolioMemo;
    private PortfolioCategoryDto portfolioCategory;
}
