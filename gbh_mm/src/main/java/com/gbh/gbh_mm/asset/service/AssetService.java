package com.gbh.gbh_mm.asset.service;

import com.gbh.gbh_mm.asset.RequestDecodeTest;
import com.gbh.gbh_mm.asset.ResponseAuthTest;
import com.gbh.gbh_mm.asset.model.vo.request.RequestCheckAccountAuth;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindCardTransactionList;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindDepositDemandTransactionList;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindDepositPayment;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindLoanPaymentList;
import com.gbh.gbh_mm.asset.model.vo.request.RequestFindSavingsPaymentList;
import com.gbh.gbh_mm.asset.model.vo.request.RequestOpenAccountAuth;
import com.gbh.gbh_mm.asset.model.vo.request.RequestWithdrawalAccountTransfer;
import com.gbh.gbh_mm.asset.model.vo.request.RequestDeleteWithdrawalAccount;
import com.gbh.gbh_mm.asset.model.vo.response.*;
import com.gbh.gbh_mm.finance.auth.vo.request.RequestCreateAccountAuth;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;

public interface AssetService {

    ResponseFindAssetList findAssetList(CustomUserDetails customUserDetails);

    ResponseFindAssetList findAssetListWithNoEncrypt(CustomUserDetails customUserDetails);

    ResponseFindDepositDemandTransactionList findDepositDemandTransactionList
        (RequestFindDepositDemandTransactionList request,
        CustomUserDetails customUserDetails);

    ResponseFindDepositPayment findDepositPayment
        (RequestFindDepositPayment request, CustomUserDetails customUserDetails);

    ResponseFindSavingsPaymentList findSavingsPaymentList(
        RequestFindSavingsPaymentList request, CustomUserDetails customUserDetails);

    ResponseFindLoanPaymentList findLoanPaymentList(
        RequestFindLoanPaymentList request,
        CustomUserDetails customUserDetails);

    ResponseFindCardTransactionList findCardTransactionList
        (RequestFindCardTransactionList request, CustomUserDetails customUserDetails);

    ResponseOpenAccountAuth openAccountAuth
        (RequestOpenAccountAuth request, CustomUserDetails customUserDetails);

    ResponseCheckAccountAuth checkAccountAuth
        (RequestCheckAccountAuth request, CustomUserDetails customUserDetails);

    ResponseFindWithdrawalAccountList findWithdrawalAccountList
        (CustomUserDetails customUserDetails);

    ResponseDeleteWithdrawalAccount deleteWithdrawalAccount(RequestDeleteWithdrawalAccount request);

    ResponseAccountTransfer accountTransfer
        (RequestWithdrawalAccountTransfer request, CustomUserDetails customUserDetails);

    ResponseAuthTest authTest();

    ResponseAuthTest decodeTest(RequestDecodeTest request);
}
