package com.gbh.gbh_mm.budget.model.entity;

import com.gbh.gbh_mm.user.model.entity.User;
import jakarta.persistence.*;
import lombok.*;


@Entity
@Getter
@Setter
@Table(name = "budget")
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Budget {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "budget_pk")
    private Long budgetPk;

    private Long budgetAmount;

    private String startDate;

    private String endDate;

    @ManyToOne(cascade = CascadeType.REMOVE)
    @JoinColumn(name = "user_pk")  // 예산이 속한 사용자와의 관계를 설정
    private User user;
}