package com.gbh.gbh_mm.user.model.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "tb_user")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "user_pk", nullable = false)
    private Long userPk;

    @Column(name = "user_name", length = 20, nullable = false)
    private String userName;

    @Column(name = "user_email")
    private String userEmail;

    @Column(name = "phone_number")
    private String phoneNumber;

    @Column(name = "birth")
    private String birth;

    @Column(name = "gender", length = 1, nullable = false)
    private char gender;

    @Column(name = "pin")
    private String pin;

    @Column(name = "connection_information")
    private String connectionInformation;


    @Column(name = "user_key")
    private String userKey;

    @Column(name = "character_image_url")
    private String characterImageUrl;

    @Column(name = "budget_feature")
    private Boolean budgetFeature;

    @Column(name = "budget_alarm_time")
    private LocalDateTime budgetAlarmTime;

    @Column(name = "salary_date")
    private Integer salaryDate;

    @Column(name = "salary_amount")
    private Long salaryAmount;

    @Column(name = "salary_account")
    private String salaryAccount;

    @Column(name = "fcm_token")
    private String fcmToken;

    @Column(name = "aes_key")
    private String aesKey;

    @Column(name = "created_at")
    private LocalDate createdAt;

    @PrePersist
    public void prePersist() {
        this.createdAt = (this.createdAt == null) ? LocalDate.now() : this.createdAt;
    }

    public void saveUserKey(String userKey) {
        this.userKey = userKey;
    }
}
