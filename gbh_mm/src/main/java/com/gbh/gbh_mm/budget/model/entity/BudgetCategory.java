package com.gbh.gbh_mm.budget.model.entity;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Getter
@Setter
@Table(name = "budgetCategory")
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BudgetCategory {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long budgetCategoryPk;

    private String budgetCategoryName;

    private Long budgetCategoryPrice = 0L;

    private Long budgetExpendAmount = 0L;

    @ManyToOne(cascade = CascadeType.REMOVE)
    @JoinColumn(name = "budget_pk")
    private Budget budget;



}
