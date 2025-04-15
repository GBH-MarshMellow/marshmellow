package com.gbh.gbh_mm.autoTransaction.service;

import com.gbh.gbh_mm.autoTransaction.model.vo.request.RequestCreateAutoTransaction;
import com.gbh.gbh_mm.autoTransaction.model.vo.response.ResponseCreateAutoTransaction;
import com.gbh.gbh_mm.autoTransaction.model.vo.response.ResponseDemandDepositList;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;

public interface AutoTransactionService {

    ResponseCreateAutoTransaction createAutoTransaction
        (RequestCreateAutoTransaction request, CustomUserDetails customUserDetails);

    ResponseDemandDepositList findDemandDepositList(CustomUserDetails customUserDetails);
}
