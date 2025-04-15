package com.gbh.gbh_mm.user.model.response;

import com.gbh.gbh_mm.user.model.dto.AccountDto;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ResponseFindAccountList {
    private List<AccountDto> accountList;
}
