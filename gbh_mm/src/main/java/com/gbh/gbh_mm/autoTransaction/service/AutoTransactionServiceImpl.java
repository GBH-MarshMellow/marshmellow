package com.gbh.gbh_mm.autoTransaction.service;

import com.gbh.gbh_mm.api.DemandDepositAPI;
import com.gbh.gbh_mm.autoTransaction.model.dto.DemandDepositDto;
import com.gbh.gbh_mm.autoTransaction.model.entity.AutoTransaction;
import com.gbh.gbh_mm.autoTransaction.model.vo.request.RequestCreateAutoTransaction;
import com.gbh.gbh_mm.autoTransaction.model.vo.response.ResponseCreateAutoTransaction;
import com.gbh.gbh_mm.autoTransaction.model.vo.response.ResponseDemandDepositList;
import com.gbh.gbh_mm.autoTransaction.repo.AutoTransactionRepository;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.repo.UserRepository;
import com.gbh.gbh_mm.wishlist.model.entity.Wishlist;
import com.gbh.gbh_mm.wishlist.repo.WishlistRepository;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class AutoTransactionServiceImpl implements AutoTransactionService {
    private final AutoTransactionRepository autoTransactionRepository;
    private final UserRepository userRepository;
    private final WishlistRepository wishlistRepository;
    private final DemandDepositAPI demandDepositAPI;

    @Override
    public ResponseCreateAutoTransaction createAutoTransaction(
        RequestCreateAutoTransaction request,
        CustomUserDetails customUserDetails
    ) {
        User user;
        Wishlist wishlist;
        try {
            user = userRepository.findById(customUserDetails.getUserPk())
                .orElseThrow(() -> new Exception());
        } catch (Exception e) {
            ResponseCreateAutoTransaction response = ResponseCreateAutoTransaction.builder()
                .message("존재하지 않는 회원입니다.")
                .build();
            return response;
        }

        try {
            wishlist = wishlistRepository.findById(request.getWishListPk())
                .orElseThrow(() -> new Exception());
        } catch (Exception e) {
            ResponseCreateAutoTransaction response = ResponseCreateAutoTransaction.builder()
                .message("존재하지 않는 위시리스트입니다.")
                .build();
            return response;
        }


        AutoTransaction autoTransaction = AutoTransaction.builder()
            .withdrawalAccountNo(request.getWithdrawalAccountNo())
            .depositAccountNo(request.getDepositAccountNo())
            .dueDate(request.getDueDate())
            .transactionBalance(request.getTransactionBalance())
            .user(user)
            .wishlist(wishlist)
            .build();

        autoTransactionRepository.save(autoTransaction);

        ResponseCreateAutoTransaction response = ResponseCreateAutoTransaction.builder()
            .message("등록 성공")
            .build();
        return response;
    }

    @Override
    public ResponseDemandDepositList findDemandDepositList(CustomUserDetails customUserDetails) {
        String userKey = customUserDetails.getUserKey();

        try {
            Map<String, Object> responseData = demandDepositAPI.findDemandDepositAccountList(userKey);
            Map<String, Object> apiData = (Map<String, Object>) responseData.get("apiResponse");
            List<Map<String, Object>> recData = (List<Map<String, Object>>) apiData.get("REC");

            List<DemandDepositDto> demandDepositDtos = new ArrayList<>();
            for (Map<String, Object> recDatum : recData) {
                DemandDepositDto demandDepositDto = DemandDepositDto.builder()
                    .bankName((String) recDatum.get("bankName"))
                    .accountNo((String) recDatum.get("accountNo"))
                    .build();

                demandDepositDtos.add(demandDepositDto);
            }

            return ResponseDemandDepositList.builder()
                .demandDepositList(demandDepositDtos)
                .build();

        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
