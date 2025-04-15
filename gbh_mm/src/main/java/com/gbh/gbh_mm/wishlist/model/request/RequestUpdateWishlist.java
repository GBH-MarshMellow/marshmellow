package com.gbh.gbh_mm.wishlist.model.request;

import lombok.Data;

@Data
public class RequestUpdateWishlist {
    private String productNickname;

    private String productName;

    private Long productPrice;

    private String productImageUrl;

    private String productUrl;

}
