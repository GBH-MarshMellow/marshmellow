package com.gbh.gbh_mm.asset.model.vo.response;

import com.gbh.gbh_mm.asset.model.dto.WithdrawalAccountDto;
import com.gbh.gbh_mm.asset.model.entity.WithdrawalAccount;
import lombok.Builder;
import lombok.Data;

import java.util.List;

@Data
@Builder
public class ResponseFindWithdrawalAccountList {
    private String iv;
    private List<WithdrawalAccountDto> withdrawalAccountList;
}
