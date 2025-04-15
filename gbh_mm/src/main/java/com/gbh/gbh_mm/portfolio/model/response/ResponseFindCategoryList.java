package com.gbh.gbh_mm.portfolio.model.response;

import com.gbh.gbh_mm.portfolio.model.dto.PortfolioCategoryDto;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ResponseFindCategoryList {
    List<PortfolioCategoryDto> portfolioCategoryList;
}
