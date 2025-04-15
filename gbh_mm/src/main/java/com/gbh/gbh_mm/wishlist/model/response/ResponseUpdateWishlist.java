package com.gbh.gbh_mm.wishlist.model.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ResponseUpdateWishlist {
    private Long wishlistPk;
    private String productNickname;
    private String productName;
    private Long productPrice;
    private Long achievePrice;
    private String productImageUrl;
    private String productUrl;
    private String isSelected;
    private String isCompleted;

}
