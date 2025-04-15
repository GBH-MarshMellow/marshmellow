package com.gbh.gbh_mm.autoTransaction.model.entity;

import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.wishlist.model.entity.Wishlist;
import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Table
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AutoTransaction {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "auto_transaction_pk")
    private int autoTransactionPk;

    @Column(name = "withdrawal_account_no")
    private String withdrawalAccountNo;

    @Column(name = "deposit_account_no")
    private String depositAccountNo;

    @Column(name = "due_date")
    private String dueDate;

    @Column(name = "transaction_balance")
    private int transactionBalance;

    @ManyToOne
    @JoinColumn(name = "user_pk")
    private User user;

    @OneToOne(cascade = CascadeType.REMOVE)
    @JoinColumn(name = "wishlist_pk")
    private Wishlist wishlist;
}
