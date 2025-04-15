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
@Table(name = "tbl_portfolio_category")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PortfolioCategory {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "portfolio_category_pk")
    private int portfolioCategoryPk;

    @Column(name = "portfolio_category_name")
    private String portfolioCategoryName;

    @Column(name = "portfolio_category_memo")
    private String portfolioCategoryMemo;

    @ManyToOne
    @JoinColumn(name = "user_pk")
    private User user;
}
