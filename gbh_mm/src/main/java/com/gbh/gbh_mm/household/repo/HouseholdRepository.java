package com.gbh.gbh_mm.household.repo;

import com.gbh.gbh_mm.household.model.entity.Household;

import com.gbh.gbh_mm.household.model.enums.HouseholdClassificationEnum;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface HouseholdRepository extends JpaRepository<Household, Long> {

    List<Household> findAllByTradeDateBetweenAndUser_UserPkOrderByTradeDateAsc
        (String startDate, String endDate, long userPk);

    List<Household> findTop2ByUser_UserPkOrderByTradeDateDesc(long userPk);

    List<Household> findAllByTradeDateBetweenAndUser_UserPkAndHouseholdMemoOrderByTradeDateAsc
        (String startDate, String endDate, long userPk, String keyword);

    @Query("SELECT h FROM Household h " +
        "WHERE h.tradeDate BETWEEN :startDate AND :endDate " +
        "AND h.user.userPk = :userPk " +
        "AND (h.householdMemo LIKE CONCAT('%', :keyword, '%') " +
        "OR h.tradeName LIKE CONCAT('%', :keyword, '%')) " +
        "ORDER BY h.tradeDate ASC")
    List<Household> searchHousehold(
        @Param("startDate") String startDate,
        @Param("endDate") String endDate,
        @Param("userPk") long userPk,
        @Param("keyword") String keyword
    );

    List<Household> findAllByTradeDateBetweenAndUser_UserPkAndHouseholdClassificationCategory
        (String startDate, String endDate, long userPk, HouseholdClassificationEnum classification);

    @Query("SELECT h FROM Household h " +
            "JOIN h.householdDetailCategory hdc " +
            "JOIN hdc.aiCategory a " +
            "WHERE h.user.userPk = :userPk " +
            "AND h.exceptedBudgetYn = 'N' " +
            "AND h.householdClassificationCategory = 'WITHDRAWAL' " +
            "AND h.tradeDate BETWEEN :startDate AND :endDate " +
            "AND a.aiCategory = :aiCategory " +
            "ORDER BY h.tradeDate DESC"
    )
    List<Household> findHouseholdsByBudget(
            @Param("userPk") long userPk,
            @Param("startDate") String startDate,
            @Param("endDate") String endDate,
            @Param("aiCategory") String aiCategory
    );

    List<Household> findAllByUser_UserPkAndHouseholdClassificationCategoryOrderByTradeDateAsc
            (Long userPk, HouseholdClassificationEnum householdClassificationEnum);

    @Query("SELECT h FROM Household h " +
            "JOIN FETCH h.householdDetailCategory dc " +
            "JOIN FETCH dc.aiCategory ac " +
            "JOIN FETCH dc.householdCategory hc " +
            "WHERE h.user.userPk = :userPk AND h.householdClassificationCategory = :classification " +
            "ORDER BY h.tradeDate ASC")
    List<Household> findAllWithDetailAndAiCategoryAndHouseholdCategory(@Param("userPk") Long userPk,
                                                                       @Param("classification") HouseholdClassificationEnum classification);


}
