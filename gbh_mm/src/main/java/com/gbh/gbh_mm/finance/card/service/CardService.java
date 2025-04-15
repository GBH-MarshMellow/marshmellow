package com.gbh.gbh_mm.finance.card.service;

import com.fasterxml.jackson.core.JsonProcessingException;
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

public interface CardService {

    Map<String, Object> findCategoryList()
        throws JsonProcessingException;

    Map<String, Object> createMerchant(RequestCreateMerchant request)
        throws JsonProcessingException;

    Map<String, Object> findCompanyList()
        throws JsonProcessingException;

    Map<String, Object> createProduct(RequestCreateCardProduct request)
        throws JsonProcessingException;

    Map<String, Object> findProductList()
        throws JsonProcessingException;

    Map<String, Object> createUserCard(RequestCreateUserCard request)
        throws JsonProcessingException;

    Map<String, Object> findUserCardList(RequestFindUserCardList request)
        throws JsonProcessingException;

    Map<String, Object> findMerchantList()
        throws JsonProcessingException;

    Map<String, Object> createTransaction(RequestCreateTransaction request)
        throws JsonProcessingException;

    Map<String, Object> findTransactionList(RequestFindCardTransactionList request)
        throws JsonProcessingException;

    Map<String, Object> deleteTransction(RequestDeleteTransaction request)
        throws JsonProcessingException;

    Map<String, Object> findBilling(RequestFindBilling request)
        throws JsonProcessingException;

    Map<String, Object> updateAccount(RequestUpdateAccount request)
        throws JsonProcessingException;
}
