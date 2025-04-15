package com.gbh.gbh_mm.household.model.vo.response;

import com.gbh.gbh_mm.household.model.dto.PaymentMethodDto;
import java.util.List;
import lombok.Data;

@Data
public class ResponsePaymentMethodList {
    private List<PaymentMethodDto> paymentMethodList;
}
