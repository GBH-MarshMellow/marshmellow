package com.gbh.gbh_cert;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/pipeline")
public class PipelineTest {

    @GetMapping
    public String test() {
        return "is WebHook Running?";
    }
}
