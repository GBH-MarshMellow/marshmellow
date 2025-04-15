package com.gbh.gbh_mm.portfolio.controller;

import com.gbh.gbh_mm.portfolio.model.entity.Portfolio;
import com.gbh.gbh_mm.portfolio.model.request.RequestCreateCategory;
import com.gbh.gbh_mm.portfolio.model.request.RequestDeleteCategory;
import com.gbh.gbh_mm.portfolio.model.request.RequestDeletePortfolio;
import com.gbh.gbh_mm.portfolio.model.request.RequestDeletePortfolioCategoryList;
import com.gbh.gbh_mm.portfolio.model.request.RequestDeletePortfolioList;
import com.gbh.gbh_mm.portfolio.model.request.RequestFindCategoryList;
import com.gbh.gbh_mm.portfolio.model.request.RequestFindPortfolio;
import com.gbh.gbh_mm.portfolio.model.request.RequestFindPortfolioList;
import com.gbh.gbh_mm.portfolio.model.request.RequestUpdateCategory;
import com.gbh.gbh_mm.portfolio.model.response.ResponseCreateCategory;
import com.gbh.gbh_mm.portfolio.model.response.ResponseCreatePortfolio;
import com.gbh.gbh_mm.portfolio.model.response.ResponseDeleteCategory;
import com.gbh.gbh_mm.portfolio.model.response.ResponseDeletePortfolio;
import com.gbh.gbh_mm.portfolio.model.response.ResponseDeletePortfolioCategoryList;
import com.gbh.gbh_mm.portfolio.model.response.ResponseDeletePortfolioList;
import com.gbh.gbh_mm.portfolio.model.response.ResponseFindCategoryList;
import com.gbh.gbh_mm.portfolio.model.response.ResponseFindPortfolio;
import com.gbh.gbh_mm.portfolio.model.response.ResponseFindPortfolioList;
import com.gbh.gbh_mm.portfolio.model.response.ResponseUpdateCategory;
import com.gbh.gbh_mm.portfolio.model.response.ResponseUpdatePortfolio;
import com.gbh.gbh_mm.portfolio.service.PortfolioService;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import lombok.AllArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/portfolio")
@AllArgsConstructor
public class PortfolioController {
    private final PortfolioService portfolioService;

    @PostMapping("/category")
    public ResponseCreateCategory createPortfolioCategory(
        @RequestBody RequestCreateCategory request,
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        ResponseCreateCategory response = portfolioService
            .createCategory(request, customUserDetails);

        return response;
    }

    @GetMapping("/category-list")
    public ResponseFindCategoryList findPortfolioCategoryList(
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        return portfolioService.findCategoryList(customUserDetails);
    }

    @DeleteMapping("/category")
    public ResponseDeleteCategory deletePortfolioCategory(
        @RequestBody RequestDeleteCategory request
    ) {
        return portfolioService.deleteCategory(request);
    }

    @PatchMapping("/category")
    public ResponseUpdateCategory updatePortfolioCategory(
        @RequestBody RequestUpdateCategory request
    ) {
        return portfolioService.updateCategory(request);
    }

    @PostMapping
    public ResponseCreatePortfolio createPortfolio(
        @AuthenticationPrincipal CustomUserDetails customUserDetails,
        @RequestParam MultipartFile file,
        @RequestParam String portfolioMemo,
        @RequestParam String fileName,
        @RequestParam int portfolioCategoryPk
    ) {
        return portfolioService
            .createPortfolio(file, portfolioMemo, fileName, customUserDetails, portfolioCategoryPk);
    }

    @GetMapping("/list")
    public ResponseFindPortfolioList findPortfolioList(
        @AuthenticationPrincipal CustomUserDetails customUserDetails
    ) {
        return portfolioService.findPortfolioList(customUserDetails);
    }

    @GetMapping
    public ResponseFindPortfolio findPortfolio(
        @RequestBody RequestFindPortfolio request
    ) {
        return portfolioService.findPortfolio(request);
    }

    @DeleteMapping
    public ResponseDeletePortfolio deletePortfolio(
        @RequestBody RequestDeletePortfolio request
    ) {
        return portfolioService.deletePortfolio(request);
    }

    @PatchMapping
    public ResponseUpdatePortfolio updatePortfolio(
        @RequestParam(required = false) MultipartFile file,
        @RequestParam String portfolioMemo,
        @RequestParam String fileName,
        @RequestParam int portfolioPk,
        @RequestParam int portfolioCategoryPk
    ) {
        return portfolioService.updatePortfolio
            (file, portfolioMemo, fileName, portfolioPk, portfolioCategoryPk);
    }

    @DeleteMapping("/category-list")
    public ResponseDeletePortfolioCategoryList deleteCategoryList(
        @AuthenticationPrincipal CustomUserDetails customUserDetails,
        @RequestBody RequestDeletePortfolioCategoryList request
    ) {
        return portfolioService.deleteCategoryList(customUserDetails, request);
    }

    @DeleteMapping("/list")
    public ResponseDeletePortfolioList deletePortfolioList(
        @AuthenticationPrincipal CustomUserDetails customUserDetails,
        @RequestBody RequestDeletePortfolioList request
    ) {
        return portfolioService.deletePortfolioList(customUserDetails, request);
    }

}
