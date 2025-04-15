package com.gbh.gbh_mm.user.model.entity;

import lombok.Getter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Collection;
import java.util.List;

@Getter
public class CustomUserDetails implements UserDetails {

    private final Long userPk;
    private final String userEmail;
    private final String phoneNumber;
    private final String birth;
    private final char gender;
    private final String pin;
    private final String userKey;
    private final String characterImageUrl;
    private final Boolean budgetFeature;
    private final LocalDateTime budgetAlarmTime;
    private final LocalDate createdAt;
    private final String aesKey;
    private final Integer salaryDate;

    public CustomUserDetails(User user) {
        this.userPk = user.getUserPk();
        this.phoneNumber = user.getPhoneNumber();
        this.userEmail = user.getUserEmail();
        this.birth = user.getBirth();
        this.pin = user.getPin();
        this.gender = user.getGender();
        this.userKey = user.getUserKey();
        this.characterImageUrl = user.getCharacterImageUrl();
        this.budgetFeature = user.getBudgetFeature();
        this.budgetAlarmTime = user.getBudgetAlarmTime();
        this.createdAt = user.getCreatedAt();
        this.aesKey = user.getAesKey();
        this.salaryDate = user.getSalaryDate();
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return List.of();
    }

    @Override
    public String getPassword() {
        return pin;
    }

    @Override
    public String getUsername() {
        return userPk.toString();
    }

    // 계정 만료, 잠금, 자격 증명 만료, 활성화 상태에 대한 메서드
    @Override
    public boolean isAccountNonExpired() {
        return true;
    }


    @Override
    public boolean isAccountNonLocked() {
        return true;
    }


    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }

}
