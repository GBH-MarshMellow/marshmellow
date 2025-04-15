package com.gbh.gbh_mm.autoTransaction.repo;

import com.gbh.gbh_mm.autoTransaction.model.entity.AutoTransaction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface AutoTransactionRepository extends JpaRepository<AutoTransaction, Integer> {

}
