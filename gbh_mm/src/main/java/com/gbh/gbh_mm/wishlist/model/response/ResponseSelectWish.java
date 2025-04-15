package com.gbh.gbh_mm.wishlist.model.response;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResponseSelectWish {
    private String message;
    private String isSelected;
}
