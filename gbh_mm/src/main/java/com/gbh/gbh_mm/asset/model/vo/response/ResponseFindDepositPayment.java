package com.gbh.gbh_mm.asset.model.vo.response;

import com.gbh.gbh_mm.asset.model.dto.DepositPaymentDto;
import lombok.Data;

import java.util.Map;

@Data
public class ResponseFindDepositPayment {
    String iv;
    private DepositPaymentDto payment;
}
