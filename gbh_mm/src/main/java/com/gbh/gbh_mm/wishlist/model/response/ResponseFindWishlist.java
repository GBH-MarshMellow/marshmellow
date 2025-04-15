package com.gbh.gbh_mm.wishlist.model.response;

import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class ResponseFindWishlist {
    private String message;
    private List<WishlistData> wishlist;

    @Data
    @Builder
    public static class WishlistData {
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
}
