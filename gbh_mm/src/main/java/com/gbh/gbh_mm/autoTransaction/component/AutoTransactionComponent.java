package com.gbh.gbh_mm.autoTransaction.component;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.DemandDepositAPI;
import com.gbh.gbh_mm.autoTransaction.model.entity.AutoTransaction;
import com.gbh.gbh_mm.autoTransaction.repo.AutoTransactionRepository;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestAccountTransfer;
import com.gbh.gbh_mm.wishlist.model.entity.Wishlist;
import com.gbh.gbh_mm.wishlist.repo.WishlistRepository;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import lombok.AllArgsConstructor;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;


@Component
@AllArgsConstructor
public class AutoTransactionComponent {

    private final AutoTransactionRepository autoTransactionRepository;
    private final DemandDepositAPI demandDepositAPI;
    private final WishlistRepository wishlistRepository;


    @Scheduled(cron = "0 0 13 * * ?")
    public void autoTransaction() {
        List<AutoTransaction> autoTransactionList = autoTransactionRepository.findAll();
        List<Wishlist> wishlists = new ArrayList<>();
        List<AutoTransaction> deleteList = new ArrayList<>();
        try {
            for (AutoTransaction autoTransaction : autoTransactionList) {
                if (autoTransaction.getWishlist().getIsSelected().equals("N")) {
                    continue;
                }

                LocalDate currentDate = LocalDate.now();

                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd");
                String currentString = currentDate.format(formatter);

                String depositSummary = "";
                String withdrawalSummary = "";
                RequestAccountTransfer request = RequestAccountTransfer.builder()
                    .depositAccountNo(autoTransaction.getDepositAccountNo())
                    .depositTransactionSummary(depositSummary)
                    .transactionBalance(autoTransaction.getTransactionBalance())
                    .withdrawalAccountNo(autoTransaction.getWithdrawalAccountNo())
                    .withdrawalTransactionSummary(withdrawalSummary)
                    .userKey(autoTransaction.getUser().getUserKey())
                    .build();
                demandDepositAPI.accountTransfer(request);

                Wishlist wishlist = autoTransaction.getWishlist();

                long productPrice = wishlist.getProductPrice();
                long achievePrice =
                    wishlist.getAchievePrice() + autoTransaction.getTransactionBalance();
                wishlist.setAchievePrice(achievePrice);

                if (productPrice == achievePrice) {
                    wishlist.setIsCompleted("Y");
                }

                wishlists.add(wishlist);

                if (autoTransaction.getDueDate().equals(currentString)
                    || productPrice == achievePrice) {
                    deleteList.add(autoTransaction);
                }
            }

            wishlistRepository.saveAll(wishlists);
            autoTransactionRepository.deleteAll(deleteList);

        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}
