package com.gbh.gbh_mm.wishlist.model.response;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResponseFindDetailWishlist {
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
