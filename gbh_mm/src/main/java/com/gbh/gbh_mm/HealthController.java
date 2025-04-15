package com.gbh.gbh_mm;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HealthController {
    @GetMapping("/health-check")
    public String healthCheck() {
        return "server is running";
    }

}
