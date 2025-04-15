package com.gbh.gbh_mm;

import com.gbh.gbh_mm.common.dto.ChildRequestAndResponse;
import com.gbh.gbh_mm.common.exception.CustomException;
import com.gbh.gbh_mm.common.exception.ErrorCode;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/test")
public class TestController {

    @PostMapping("/child")
    public ChildRequestAndResponse getChild(@RequestBody ChildRequestAndResponse requestAndResponse) {
        int childId = requestAndResponse.getChildId();

        // childId가 1이면 성공, 아니면 예외 발생
        if (childId == 1) {
            return requestAndResponse;
        } else {
            throw new CustomException(ErrorCode.CHILD_NOT_FOUND);
        }

    }

}