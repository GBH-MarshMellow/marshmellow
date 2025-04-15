package com.gbh.gbh_mm.asset.model.vo.response;

import com.gbh.gbh_mm.asset.model.dto.SavingsPaymentDto;
import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class ResponseFindSavingsPaymentList {
    private String iv;
    private SavingsPaymentDto paymentList;
}
