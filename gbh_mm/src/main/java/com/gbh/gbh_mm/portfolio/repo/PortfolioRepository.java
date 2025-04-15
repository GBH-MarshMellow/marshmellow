package com.gbh.gbh_mm.portfolio.repo;

import com.gbh.gbh_mm.portfolio.model.entity.Portfolio;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface PortfolioRepository extends JpaRepository<Portfolio, Integer> {

    List<Portfolio> findAllByUser_UserPk(Long userUserPk);

    List<Portfolio> findByPortfolioCategory_PortfolioCategoryPk(int categoryPk);

    void deleteAllByPortfolioCategory_PortfolioCategoryPk(int portfolioCategoryPortfolioCategoryPk);
}
