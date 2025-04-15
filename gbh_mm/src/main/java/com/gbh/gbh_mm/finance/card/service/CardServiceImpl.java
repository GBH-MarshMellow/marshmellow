package com.gbh.gbh_mm.finance.card.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.CardAPI;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateCardProduct;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateMerchant;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateTransaction;
import com.gbh.gbh_mm.finance.card.vo.request.RequestCreateUserCard;
import com.gbh.gbh_mm.finance.card.vo.request.RequestDeleteTransaction;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindBilling;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindCardTransactionList;
import com.gbh.gbh_mm.finance.card.vo.request.RequestFindUserCardList;
import com.gbh.gbh_mm.finance.card.vo.request.RequestUpdateAccount;
import java.util.Map;
import lombok.AllArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@AllArgsConstructor
public class CardServiceImpl implements CardService {

    private final CardAPI cardAPI;

    @Override
    public Map<String, Object> findCategoryList() throws JsonProcessingException {
        return cardAPI.findCategoryList();
    }

    @Override
    public Map<String, Object> createMerchant(RequestCreateMerchant request)
        throws JsonProcessingException {
        return cardAPI.createMerchant(request);
    }

    @Override
    public Map<String, Object> findCompanyList() throws JsonProcessingException {
        return cardAPI.findCompanyList();
    }

    @Override
    public Map<String, Object> createProduct(RequestCreateCardProduct request)
        throws JsonProcessingException {
        return cardAPI.createProduct(request);
    }

    @Override
    public Map<String, Object> findProductList() throws JsonProcessingException {
        return cardAPI.findProductList();
    }

    @Override
    public Map<String, Object> createUserCard(RequestCreateUserCard request)
        throws JsonProcessingException {
        return cardAPI.createUserCard(request);
    }

    @Override
    public Map<String, Object> findUserCardList(RequestFindUserCardList request)
        throws JsonProcessingException {
        return cardAPI.findUserCardList(request.getUserKey());
    }

    @Override
    public Map<String, Object> findMerchantList() throws JsonProcessingException {
        return cardAPI.findMerchantList();
    }

    @Override
    public Map<String, Object> createTransaction(RequestCreateTransaction request)
        throws JsonProcessingException {
        return cardAPI.createTransaction(request);
    }

    @Override
    public Map<String, Object> findTransactionList(RequestFindCardTransactionList request)
        throws JsonProcessingException {
        return cardAPI.findTransactionList(request);
    }

    @Override
    public Map<String, Object> deleteTransction(RequestDeleteTransaction request)
        throws JsonProcessingException {
        return cardAPI.deleteTransaction(request);
    }

    @Override
    public Map<String, Object> findBilling(RequestFindBilling request)
        throws JsonProcessingException {
        return cardAPI.findBilling(request);
    }

    @Override
    public Map<String, Object> updateAccount(RequestUpdateAccount request)
        throws JsonProcessingException {
        return cardAPI.updateAccount(request);
    }
}
