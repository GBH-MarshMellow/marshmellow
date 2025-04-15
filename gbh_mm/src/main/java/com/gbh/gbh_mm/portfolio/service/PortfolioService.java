package com.gbh.gbh_mm.portfolio.service;

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
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.multipart.MultipartFile;

public interface PortfolioService {

    ResponseCreateCategory createCategory(RequestCreateCategory request,
        CustomUserDetails customUserDetails);

    ResponseFindCategoryList findCategoryList(CustomUserDetails customUserDetails);

    ResponseDeleteCategory deleteCategory(RequestDeleteCategory request);

    ResponseUpdateCategory updateCategory(RequestUpdateCategory request);

    ResponseCreatePortfolio createPortfolio
        (MultipartFile file, String portfolioMemo, String fileName,
            CustomUserDetails customUserDetails, int portfolioCategoryPk);

    ResponseFindPortfolioList findPortfolioList(CustomUserDetails customUserDetails);

    ResponseFindPortfolio findPortfolio(RequestFindPortfolio request);

    ResponseDeletePortfolio deletePortfolio(RequestDeletePortfolio request);

    ResponseUpdatePortfolio updatePortfolio(MultipartFile file, String portfolioMemo, String fileName, int portfolioPk, int portfolioCategoryPk);

    ResponseDeletePortfolioCategoryList deleteCategoryList
        (CustomUserDetails customUserDetails, RequestDeletePortfolioCategoryList request);

    ResponseDeletePortfolioList deletePortfolioList(CustomUserDetails customUserDetails, RequestDeletePortfolioList request);
}
