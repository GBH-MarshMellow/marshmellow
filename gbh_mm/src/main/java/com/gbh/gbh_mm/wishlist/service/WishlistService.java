package com.gbh.gbh_mm.wishlist.service;

import com.gbh.gbh_mm.common.exception.CustomException;
import com.gbh.gbh_mm.common.exception.ErrorCode;
import com.gbh.gbh_mm.s3.S3Component;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.repo.UserRepository;
import com.gbh.gbh_mm.wishlist.model.entity.Wishlist;
import com.gbh.gbh_mm.wishlist.model.request.RequestSelectWish;
import com.gbh.gbh_mm.wishlist.model.request.RequestUpdateWishlist;
import com.gbh.gbh_mm.wishlist.model.response.*;
import com.gbh.gbh_mm.wishlist.repo.WishlistRepository;
import io.github.bonigarcia.wdm.WebDriverManager;
import lombok.RequiredArgsConstructor;
import org.modelmapper.ModelMapper;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.net.URISyntaxException;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.springframework.web.multipart.MultipartFile;

@Service
@RequiredArgsConstructor
public class WishlistService {

    private final WishlistRepository wishlistRepository;
    private final UserRepository userRepository;
    private final S3Component s3Component;
    private final ModelMapper mapper;

    // 위시리스트 생성
    @Transactional
    public ResponseCreateWishlist createWishlist(
            Long userPk, String productNickname, String productName, Long productPrice,
            String productUrl, MultipartFile file
    ) {
        User user = userRepository.findById(userPk)
                .orElseThrow(() -> new CustomException(ErrorCode.CHILD_NOT_FOUND));

        String fileUrl = s3Component.uploadFile(file);

        Wishlist wishlist = Wishlist.builder()
                .achievePrice(0L)
                .productNickname(productNickname)
                .productName(productName)
                .productPrice(productPrice)
                .productUrl(productUrl)
                .isSelected("N")
                .isCompleted("N")
                .productImageUrl(fileUrl)
                .user(user)
                .build();

        Wishlist savedWishList = wishlistRepository.save(wishlist);
        ResponseCreateWishlist response = mapper.map(savedWishList, ResponseCreateWishlist.class);

        return response;
    }

    // 위시리스트 조회
    public ResponseFindWishlist getWishlist(Long userPk) {
        List<Wishlist> wishlist = wishlistRepository.findAllByUser_UserPk(userPk);
        List<ResponseFindWishlist.WishlistData> wishlistData = wishlist.stream()
                .map(wish -> ResponseFindWishlist.WishlistData.builder()
                        .wishlistPk(wish.getWishlistPk())
                        .productNickname(wish.getProductNickname())
                        .productName(wish.getProductName())
                        .productPrice(wish.getProductPrice())
                        .achievePrice(wish.getAchievePrice())
                        .productImageUrl(wish.getProductImageUrl())
                        .productUrl(wish.getProductUrl())
                        .isSelected(wish.getIsSelected())
                        .isCompleted(wish.getIsCompleted())
                        .build()
                )
                .collect(Collectors.toList());

        return ResponseFindWishlist.builder()
                .message("위시리스트 조회")
                .wishlist(wishlistData)
                .build();


    }

    // 위시리스트 상세 조회
    public ResponseFindDetailWishlist getWishlistDetail(Long wishlistPk) {

        Wishlist wishlist = wishlistRepository.findById(wishlistPk)
                .orElseThrow(() -> new CustomException(ErrorCode.RESOURCE_NOT_FOUND));

        return ResponseFindDetailWishlist.builder()
                .wishlistPk(wishlist.getWishlistPk())
                .productNickname(wishlist.getProductNickname())
                .productName(wishlist.getProductName())
                .productPrice(wishlist.getProductPrice())
                .achievePrice(wishlist.getAchievePrice())
                .productImageUrl(wishlist.getProductImageUrl())
                .productUrl(wishlist.getProductUrl())
                .isSelected(wishlist.getIsSelected())
                .isCompleted(wishlist.getIsCompleted())
                .build();
    }

    // 위시리스트 수정
    public ResponseUpdateWishlist updateWishlist(Long wishlistPk, String productNickname, String productName,
                                                 Long productPrice, String productUrl, MultipartFile file) {
        Wishlist wishlist = wishlistRepository.findById(wishlistPk)
                .orElseThrow(() -> new CustomException(ErrorCode.RESOURCE_NOT_FOUND));


        if (!productNickname.isEmpty() && productNickname != null) {
            wishlist.setProductNickname(productNickname);
        }

        if (!productName.isEmpty() && productName != null) {
            wishlist.setProductName(productName);
        }

        if (productPrice != 0 && productPrice != null) {
            wishlist.setProductPrice(productPrice);
        }

        if (!productUrl.isEmpty() && productUrl != null) {
            wishlist.setProductUrl(productUrl);
        }

        if (!file.isEmpty() && file != null) {
            try {
                s3Component.deleteFileByUrl(wishlist.getProductImageUrl());
                String fileUrl = s3Component.uploadFile(file);
                wishlist.setProductImageUrl(fileUrl);
            } catch (URISyntaxException e) {
                throw new RuntimeException(e);
            }
        }

        Wishlist updatedWishList = wishlistRepository.save(wishlist);



        return mapper.map(updatedWishList, ResponseUpdateWishlist.class);
    }

    // 위시 선택
    public ResponseSelectWish selectWish(Long wishlistPk , RequestSelectWish requestSelectWish) {
        Wishlist wish = wishlistRepository.findById(wishlistPk).orElseThrow(() -> new CustomException(ErrorCode.RESOURCE_NOT_FOUND));

        wish.setIsSelected(requestSelectWish.getIsSelected());
        wishlistRepository.save(wish);
        return ResponseSelectWish.builder()
                .message("위시 pk " + wishlistPk +" 수정 완료" )
                .isSelected(requestSelectWish.getIsSelected())
                .build();
    }

    // 위시리스트 삭제
    @Transactional
    public ResponseDeleteWishlist deleteWishlist(Long wishlistPk) {
        Wishlist wishlist = wishlistRepository.findById(wishlistPk)
                .orElseThrow(() -> new CustomException(ErrorCode.RESOURCE_NOT_FOUND));

        wishlistRepository.delete(wishlist);
        return ResponseDeleteWishlist.builder()
                .message("위시리스트 삭제 완료")
                .deleteWishlistPk(wishlistPk)
                .build();
    }

    // 현재 위시 조회
    public ResponseFindDetailWishlist getCurrentWish(Long userPk) {
        List<Wishlist> wishlist = wishlistRepository.findAllByUser_UserPk(userPk)
                .stream()
                .filter(wish ->
                        wish.getIsSelected().equals("Y") && wish.getIsCompleted().equals("N")
                )
                .collect(Collectors.toList());

        if (wishlist.isEmpty()) {
            throw new CustomException(ErrorCode.RESOURCE_NOT_FOUND);
        }

        if (wishlist.size() > 1) {
            throw new CustomException(ErrorCode.DATABASE_ERROR);
        }

        Wishlist wish = wishlist.get(0);
        return ResponseFindDetailWishlist.builder()
                .wishlistPk(wish.getWishlistPk())
                .productNickname(wish.getProductNickname())
                .productName(wish.getProductName())
                .productPrice(wish.getProductPrice())
                .achievePrice(wish.getAchievePrice())
                .productImageUrl(wish.getProductImageUrl())
                .productUrl(wish.getProductUrl())
                .isSelected(wish.getIsSelected())
                .isCompleted(wish.getIsCompleted())
                .build();
    }

    // 링크 Jsoup
    public ResponseJsoupLink jsoupLink(String url) {

        // ChromeDriver 자동 다운로드
        WebDriverManager.chromedriver().setup();

        ChromeOptions options = new ChromeOptions();
        options.addArguments("--headless=new");
        options.addArguments("--no-sandbox");
        options.addArguments("--disable-dev-shm-usage");
        options.addArguments("--remote-allow-origins=*");

        // Chrome 브라우저 실행
        WebDriver driver = new ChromeDriver(options);
        driver.get(url);

        // 로딩 대기 (봇 감지 우회를 위해 추가)
        driver.manage().timeouts().implicitlyWait(5, TimeUnit.SECONDS);

        // OG 태그 가져오기
        String ogTitle = getMetaTagContent(driver, "og:title");
        String ogImage = getMetaTagContent(driver, "og:image");

        // 결과 출력
        System.out.println("OG Title: " + ogTitle);
        System.out.println("OG Image: " + ogImage);

        // 브라우저 종료
        driver.quit();

        return ResponseJsoupLink.builder()
                .message("크롤링 완료")
                .productName(ogTitle)
                .productImage(ogImage)
                .build();
    }

    // OG 태그의 content 값 가져오는 함수
    public static String getMetaTagContent(WebDriver driver, String property) {
        List<WebElement> metaTags = driver.findElements(By.cssSelector("meta[property='" + property + "']"));

        if (!metaTags.isEmpty()) {
            return metaTags.get(0).getAttribute("content");
        } else {
            return "Meta 태그를 찾을 수 없습니다";
        }
    }

}
