package com.gbh.gbh_mm.wishlist.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ResponseCreateWishlist {
    private String message;

    private Long wishlistPk;

    private String productNickname;

    private String productName;

    private Long productPrice;

    private String productImageUrl;

    private String productUrl;

    private String isSelected;

    private String isCompleted;
}
