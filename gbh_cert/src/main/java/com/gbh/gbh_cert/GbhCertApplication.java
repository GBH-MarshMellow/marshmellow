package com.gbh.gbh_cert;

import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.security.Security;

@SpringBootApplication
public class GbhCertApplication {
    static{
        // 보안 프로바이더 설정
        Security.addProvider(new BouncyCastleProvider());
    }
    public static void main(String[] args) {
        SpringApplication.run(GbhCertApplication.class, args);
    }

}
