package com.gbh.gbh_mm.wishlist.model.response;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResponseJsoupLink {
    private String message;

    private String productName;

    private String productImage;

//    private int productPrice;
}
