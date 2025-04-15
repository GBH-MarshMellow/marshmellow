package com.gbh.gbh_mm.portfolio.repo;

import com.gbh.gbh_mm.portfolio.model.entity.PortfolioCategory;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PortfolioCategoryRepository extends JpaRepository<PortfolioCategory, Integer> {

    List<PortfolioCategory> findAllByUser_UserPk(long userPk);

    PortfolioCategory findByUser_UserPkAndPortfolioCategoryName(Long userPk, String 미분류);
}
