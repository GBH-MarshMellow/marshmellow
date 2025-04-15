package com.gbh.gbh_mm.wishlist.controller;

import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import com.gbh.gbh_mm.wishlist.model.entity.Wishlist;
import com.gbh.gbh_mm.wishlist.model.request.RequestJsoupLink;
import com.gbh.gbh_mm.wishlist.model.request.RequestSelectWish;
import com.gbh.gbh_mm.wishlist.model.request.RequestUpdateWishlist;
import com.gbh.gbh_mm.wishlist.model.response.*;
import com.gbh.gbh_mm.wishlist.service.WishlistService;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("mm/wishlist")
@RequiredArgsConstructor
public class WishlistController {

    private final WishlistService wishlistService;

    // 위시리스트 생성
    @PostMapping
    public ResponseCreateWishlist createWishlist(
            @AuthenticationPrincipal CustomUserDetails userDetails,
            @RequestParam String productNickname,
            @RequestParam String productName,
            @RequestParam Long productPrice,
            @RequestParam String productUrl,
            @RequestParam MultipartFile file
    ) {
        return wishlistService
                .createWishlist(userDetails.getUserPk(), productNickname,
                        productName, productPrice, productUrl, file);
    }

    // 위시리스트 조회
    @GetMapping
    public ResponseFindWishlist getWishlist(
            @AuthenticationPrincipal CustomUserDetails userDetails) {
        return wishlistService.getWishlist(userDetails.getUserPk());
    }

    // 위시리스트 상세 조회
    @GetMapping("/detail/{wishlistPk}")
    public ResponseFindDetailWishlist getWishlistDetail(@PathVariable Long wishlistPk) {
        return wishlistService.getWishlistDetail(wishlistPk);

    }

    // 위시리스트 수정
    @PutMapping("/detail/{wishlistPk}")
    public ResponseUpdateWishlist updateWishlist(
            @PathVariable Long wishlistPk,
            @RequestParam String productNickname,
            @RequestParam String productName,
            @RequestParam Long productPrice,
            @RequestParam String productUrl,
            @RequestParam MultipartFile file
    ) {
        return wishlistService.updateWishlist(wishlistPk, productNickname, productName,
                productPrice, productUrl, file);
    }

    // 위시 선택
    @PostMapping("/detail/{wishlistPk}")
    public ResponseSelectWish selectWish(@PathVariable Long wishlistPk, @RequestBody RequestSelectWish requestSelectWish) {
        return wishlistService.selectWish(wishlistPk, requestSelectWish);
    }

    // 위시리스트 삭제
    @DeleteMapping("/detail/{wishlistPk}")
    public ResponseDeleteWishlist deleteWishlist(@PathVariable Long wishlistPk) {
        return wishlistService.deleteWishlist(wishlistPk);
    }

    // 현재 위시 조회
    @GetMapping("/detail")
    public ResponseFindDetailWishlist getCurrentWish(
            @AuthenticationPrincipal CustomUserDetails userDetails) {
        return wishlistService.getCurrentWish(userDetails.getUserPk());
    }

    // 링크로 크롤링
    @PostMapping("/jsoup")
    public ResponseJsoupLink jsoupLink(@RequestBody RequestJsoupLink requestJsoupLink) {
        return wishlistService.jsoupLink(requestJsoupLink.getUrl());
    }
}