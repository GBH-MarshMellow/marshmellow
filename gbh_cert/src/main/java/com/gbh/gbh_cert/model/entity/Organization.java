package com.gbh.gbh_cert.model.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "tb_organization")
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Organization {
    @Id
    @Column(name = "org_code", nullable = false)
    private String orgCode;

    @Column(name = "org_name")
    private String orgName;

    @Enumerated(EnumType.STRING)
    @Column(name = "org_type")
    private OrgType orgType;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    public enum OrgType {
        BANK, CARD
    }

}
