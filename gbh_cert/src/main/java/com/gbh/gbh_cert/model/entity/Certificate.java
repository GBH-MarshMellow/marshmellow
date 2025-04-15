package com.gbh.gbh_cert.model.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "tb_certificate")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Certificate {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "cert_id", nullable = false)
    private Long certId;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "cert_serial")
    private Long certSerial;

    @Column(name = "cert_data", columnDefinition = "TEXT")
    private String certData;

    @Enumerated(EnumType.STRING)
    @Column(name = "cert_status")
    private CertStatus certStatus;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @Column(name = "expired_at")
    private LocalDateTime expiredAt;

    public enum CertStatus {
        VALID, EXPIRED, REVOKED
    }
    @PrePersist
    public void prePersist() {
        this.createdAt = (this.createdAt == null) ? LocalDateTime.now() : this.createdAt;
    }
}
