package com.gbh.gbh_mm.asset.model.entity;

import com.gbh.gbh_mm.user.model.entity.User;
import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "tbl_withdrawal_account")
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Data
public class WithdrawalAccount {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "withdrawal_account_id")
    private int withdrawalAccountId;

    @Column(name = "account_no")
    private String accountNo;

    @ManyToOne
    @JoinColumn(name = "user_id")
    private User user;
}
