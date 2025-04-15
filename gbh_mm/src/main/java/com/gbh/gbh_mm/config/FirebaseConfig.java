package com.gbh.gbh_mm.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

import javax.annotation.PostConstruct;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

@Configuration
public class FirebaseConfig {

    // application.yml에 상대 경로로 지정 (예: gbh-mm-firebase.json)
    @Value("${firebase.service.account.path}")
    private String firebaseServiceAccountPath;

    @PostConstruct
    public void initialize() {
        try (InputStream serviceAccount = getClass().getClassLoader().getResourceAsStream(firebaseServiceAccountPath)) {
//        try (FileInputStream serviceAccount = new FileInputStream(firebaseServiceAccountPath)) {
            if (serviceAccount == null) {
                throw new IOException("Resource not found: " + firebaseServiceAccountPath);
            }
            FirebaseOptions options = FirebaseOptions.builder()
                .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                .build();

            if (FirebaseApp.getApps().isEmpty()) {
                FirebaseApp.initializeApp(options);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Firebase 초기화 실패");
        }
    }
}
