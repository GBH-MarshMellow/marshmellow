package com.gbh.gbh_mm.wishlist.model.entity;

import com.gbh.gbh_mm.user.model.entity.User;
import jakarta.persistence.*;
import lombok.*;

@Entity
@Getter
@Setter
@Table(name = "wishlist")
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Wishlist {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long wishlistPk;

    private String productNickname;

    private String productName;

    private Long productPrice;

    private Long achievePrice = 0L;

    private String productImageUrl;

    private String productUrl;

    private String isSelected = "N";

    private String isCompleted = "N";

    @ManyToOne
    @JoinColumn(name = "user_pk")
    private User user;
}
