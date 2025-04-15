package com.gbh.gbh_mm.portfolio.model.dto;

import lombok.Data;

@Data
public class PortfolioDto {
    private int portfolioPk;
    private String fileUrl;
    private String createDate;
    private String createTime;
    private String originFileName;
    private String fileName;
    private String portfolioMemo;
    private PortfolioCategoryDto portfolioCategory;
}
