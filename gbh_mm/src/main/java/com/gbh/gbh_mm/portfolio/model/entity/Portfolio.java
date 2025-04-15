package com.gbh.gbh_mm.portfolio.model.entity;

import com.gbh.gbh_mm.user.model.entity.User;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table(name = "tbl_portfolio")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Portfolio {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "portfolio_pk")
    private int portfolioPk;

    @Column(name = "file_url")
    private String fileUrl;

    @Column(name = "craete_date")
    private String createDate;

    @Column(name = "create_time")
    private String createTime;

    @Column(name = "origin_file_name")
    private String originFileName;

    @Column(name = "file_name")
    private String fileName;

    @Column(name = "portfolio_memo")
    private String portfolioMemo;

    @ManyToOne
    @JoinColumn(name = "user_pk")
    private User user;

    @ManyToOne
    @JoinColumn(name = "portfolio_category_pk")
    private PortfolioCategory portfolioCategory;
}
