package com.gbh.gbh_mm.portfolio.service;

import com.gbh.gbh_mm.common.exception.CustomException;
import com.gbh.gbh_mm.common.exception.ErrorCode;
import com.gbh.gbh_mm.portfolio.model.dto.PortfolioCategoryDto;
import com.gbh.gbh_mm.portfolio.model.dto.PortfolioDto;
import com.gbh.gbh_mm.portfolio.model.entity.Portfolio;
import com.gbh.gbh_mm.portfolio.model.entity.PortfolioCategory;
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
import com.gbh.gbh_mm.portfolio.repo.PortfolioCategoryRepository;
import com.gbh.gbh_mm.portfolio.repo.PortfolioRepository;
import com.gbh.gbh_mm.s3.S3Component;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.repo.UserRepository;
import jakarta.persistence.EntityNotFoundException;
import java.net.URISyntaxException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import lombok.AllArgsConstructor;
import org.modelmapper.ModelMapper;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

@Service
@AllArgsConstructor
public class PortfolioServiceImpl implements PortfolioService {

    private final PortfolioRepository portfolioRepository;
    private final PortfolioCategoryRepository portfolioCategoryRepository;
    private final UserRepository userRepository;
    private final ModelMapper mapper;
    private final S3Component s3Component;

    @Override
    public ResponseCreateCategory createCategory(RequestCreateCategory request,
        CustomUserDetails customUserDetails) {
        User user = userRepository.findByUserPk(customUserDetails.getUserPk())
            .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

        PortfolioCategory portfolioCategory = PortfolioCategory.builder()
            .portfolioCategoryMemo(request.getCategoryMemo())
            .portfolioCategoryName(request.getCategoryName())
            .user(user)
            .build();

        portfolioCategoryRepository.save(portfolioCategory);
        List<PortfolioCategory> portfolioCategoryList = portfolioCategoryRepository
            .findAllByUser_UserPk(customUserDetails.getUserPk());

        List<PortfolioCategoryDto> portfolioCategoryDtoList = portfolioCategoryList.stream()
            .map(p -> mapper.map(p, PortfolioCategoryDto.class))
            .collect(Collectors.toList());

        ResponseCreateCategory response = ResponseCreateCategory.builder()
            .portfolioCategoryList(portfolioCategoryDtoList)
            .build();

        return response;
    }

    @Override
    public ResponseFindCategoryList findCategoryList(CustomUserDetails customUserDetails) {
        List<PortfolioCategory> portfolioCategoryList = portfolioCategoryRepository
            .findAllByUser_UserPk(customUserDetails.getUserPk());

        List<PortfolioCategoryDto> portfolioCategoryDtoList = portfolioCategoryList.stream()
            .map(p -> mapper.map(p, PortfolioCategoryDto.class))
            .collect(Collectors.toList());

        ResponseFindCategoryList response = ResponseFindCategoryList.builder()
            .portfolioCategoryList(portfolioCategoryDtoList)
            .build();

        return response;
    }

    @Override
    public ResponseDeleteCategory deleteCategory(RequestDeleteCategory request) {
        try {
            List<Portfolio> portfolioList =
                portfolioRepository.findByPortfolioCategory_PortfolioCategoryPk(
                    request.getCategoryPk());

            portfolioRepository.deleteAll(portfolioList);

            portfolioCategoryRepository.deleteById(request.getCategoryPk());

            ResponseDeleteCategory response = ResponseDeleteCategory.builder()
                .message("SUCCESS")
                .build();

            return response;

        } catch (Exception e) {
            ResponseDeleteCategory response = ResponseDeleteCategory.builder()
                .message("FAIL")
                .build();

            return response;
        }
    }

    @Override
    public ResponseUpdateCategory updateCategory(RequestUpdateCategory request) {
        PortfolioCategory portfolioCategory = portfolioCategoryRepository
            .findById(request.getCategoryPk())
            .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 카테고리"));

        if (request.getCategoryMemo() != null) {
            portfolioCategory.setPortfolioCategoryMemo(request.getCategoryMemo());
        }

        if (request.getCategoryName() != null) {
            portfolioCategory.setPortfolioCategoryName(request.getCategoryName());
        }

        portfolioCategoryRepository.save(portfolioCategory);

        ResponseUpdateCategory response = mapper.map(portfolioCategory,
            ResponseUpdateCategory.class);

        return response;
    }

    @Override
    public ResponseCreatePortfolio createPortfolio(MultipartFile file, String portfolioMemo,
        String fileName, CustomUserDetails customUserDetails, int portfolioCategoryPk) {
        User user = userRepository.findByUserPk(customUserDetails.getUserPk())
            .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

        String fileUrl = s3Component.uploadFile(file);

        LocalDateTime now = LocalDateTime.now();
        String date = now.format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        String time = now.format(DateTimeFormatter.ofPattern("HHmm"));

        Portfolio portfolio = Portfolio.builder()
            .fileUrl(fileUrl)
            .fileName(fileName)
            .createDate(date)
            .createTime(time)
            .originFileName(file.getOriginalFilename())
            .fileName(fileName)
            .portfolioMemo(portfolioMemo)
            .user(user)
            .build();

        try {
            PortfolioCategory portfolioCategory = portfolioCategoryRepository
                .findById(portfolioCategoryPk)
                .orElseThrow(() -> new EntityNotFoundException());
            portfolio.setPortfolioCategory(portfolioCategory);
        } catch (EntityNotFoundException e) {
            PortfolioCategory portfolioCategory = portfolioCategoryRepository
                .findByUser_UserPkAndPortfolioCategoryName(customUserDetails.getUserPk(), "미분류");

            if (portfolioCategory == null) {
                PortfolioCategory newCategory = PortfolioCategory.builder()
                    .portfolioCategoryMemo("")
                    .portfolioCategoryName("미분류")
                    .user(user)
                    .build();
                PortfolioCategory savedCategory = portfolioCategoryRepository.save(newCategory);
                portfolio.setPortfolioCategory(savedCategory);
            } else {
                portfolio.setPortfolioCategory(portfolioCategory);
            }
        }

        Portfolio savedPortfolio = portfolioRepository.save(portfolio);

        PortfolioCategoryDto portfolioCategoryDto =
            mapper.map(savedPortfolio.getPortfolioCategory(), PortfolioCategoryDto.class);

        ResponseCreatePortfolio response = mapper.map(savedPortfolio,
            ResponseCreatePortfolio.class);
        response.setPortfolioCategory(portfolioCategoryDto);

        return response;
    }

    @Override
    public ResponseFindPortfolioList findPortfolioList(CustomUserDetails customUserDetails) {
        List<Portfolio> portfolioList = portfolioRepository
            .findAllByUser_UserPk(customUserDetails.getUserPk());

        List<PortfolioDto> portfolioDtoList = new ArrayList<>();

        for (Portfolio portfolio : portfolioList) {
            PortfolioCategoryDto portfolioCategoryDto =
                mapper.map(portfolio.getPortfolioCategory(), PortfolioCategoryDto.class);

            PortfolioDto portfolioDto = mapper.map(portfolio, PortfolioDto.class);
            portfolioDto.setPortfolioCategory(portfolioCategoryDto);

            portfolioDtoList.add(portfolioDto);
        }

        ResponseFindPortfolioList response = ResponseFindPortfolioList.builder()
            .portfolioList(portfolioDtoList)
            .build();

        return response;
    }

    @Override
    public ResponseFindPortfolio findPortfolio(RequestFindPortfolio request) {
        Portfolio portfolio = portfolioRepository.findById(request.getPortfolioPk())
            .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 포트폴리오"));

        PortfolioCategoryDto portfolioCategoryDto =
            mapper.map(portfolio.getPortfolioCategory(), PortfolioCategoryDto.class);

        ResponseFindPortfolio response = mapper.map(portfolio, ResponseFindPortfolio.class);
        response.setPortfolioCategory(portfolioCategoryDto);

        return response;
    }

    @Override
    public ResponseDeletePortfolio deletePortfolio(RequestDeletePortfolio request) {
        Portfolio portfolio = portfolioRepository.findById(request.getPortfolioPk())
            .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 포트폴리오"));

        ResponseDeletePortfolio response = new ResponseDeletePortfolio();
        String fileUrl = portfolio.getFileUrl();

        try {
            s3Component.deleteFileByUrl(fileUrl);

            portfolioRepository.delete(portfolio);

            response.setMessage("SUCCESS");

            return response;

        } catch (URISyntaxException e) {
            response.setMessage("FAIL");

            return response;
        } catch (Exception e) {
            response.setMessage("FAIL");

            return response;
        }
    }

    @Override
    public ResponseUpdatePortfolio updatePortfolio(MultipartFile file, String portfolioMemo,
        String fileName, int portfolioPk, int portfolioCategoryPk) {
        Portfolio portfolio = portfolioRepository.findById(portfolioPk)
            .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 포폴"));
        PortfolioCategory portfolioCategory = portfolioCategoryRepository
            .findById(portfolioCategoryPk)
            .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 카테고리"));

        portfolio.setPortfolioCategory(portfolioCategory);

        if (file != null && !file.isEmpty()) {
            try {
                s3Component.deleteFileByUrl(portfolio.getFileUrl());
                String fileUrl = s3Component.uploadFile(file);
                portfolio.setFileUrl(fileUrl);
                portfolio.setOriginFileName(file.getOriginalFilename());
            } catch (URISyntaxException e) {
                throw new RuntimeException(e);
            }
        }
        if (portfolioMemo != null && !portfolioMemo.isEmpty()) {
            portfolio.setPortfolioMemo(portfolioMemo);
        }
        if (fileName != null && !fileName.isEmpty()) {
            portfolio.setFileName(fileName);
        }

        portfolioRepository.save(portfolio);

        ResponseUpdatePortfolio response = mapper.map(portfolio, ResponseUpdatePortfolio.class);
        PortfolioCategoryDto portfolioCategoryDto =
            mapper.map(portfolio.getPortfolioCategory(), PortfolioCategoryDto.class);
        response.setPortfolioCategory(portfolioCategoryDto);

        return response;
    }

    @Override
    @Transactional
    public ResponseDeletePortfolioCategoryList deleteCategoryList(
        CustomUserDetails customUserDetails, RequestDeletePortfolioCategoryList request) {

        List<Integer> categoryPkList = request.getPortfolioCategoryPkList();
        try {
            for (Integer i : categoryPkList) {
                PortfolioCategory portfolioCategory = portfolioCategoryRepository.findById(i)
                    .orElseThrow(() -> new EntityNotFoundException("존재하지 않는 카테고리"));

                portfolioRepository.deleteAllByPortfolioCategory_PortfolioCategoryPk(i);

                portfolioCategoryRepository.delete(portfolioCategory);
            }

            ResponseDeletePortfolioCategoryList response = ResponseDeletePortfolioCategoryList.builder()
                .message("SUCCESS")
                .build();

            return response;
        } catch (Exception e) {
            e.printStackTrace();

            ResponseDeletePortfolioCategoryList response = ResponseDeletePortfolioCategoryList.builder()
                .message("FAIL")
                .build();

            return response;
        }
    }

    @Override
    @Transactional
    public ResponseDeletePortfolioList deletePortfolioList(
        CustomUserDetails customUserDetails,
        RequestDeletePortfolioList request
    ) {
        try {
            List<Integer> portfolioPkList = request.getPortfolioPkList();
            portfolioRepository.deleteAllById(portfolioPkList);

            ResponseDeletePortfolioList response = ResponseDeletePortfolioList.builder()
                .message("SUCCESS")
                .build();

            return response;
        } catch (Exception e) {
            e.printStackTrace();

            ResponseDeletePortfolioList response = ResponseDeletePortfolioList.builder()
                .message("FAIL")
                .build();

            return response;
        }

    }
}
