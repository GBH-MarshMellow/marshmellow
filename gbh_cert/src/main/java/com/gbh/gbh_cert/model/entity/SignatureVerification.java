package com.gbh.gbh_cert.model.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "tb_signature_verification")
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SignatureVerification {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "verification_id", nullable = false)
    private Long verificationId;

    @ManyToOne
    @JoinColumn(name = "signature_req_id", nullable = false)
    private SignatureRequest signatureRequest;

    @Column(name = "signature_verified")
    private Boolean signatureVerified;

    @Column(name = "verified_at")
    private LocalDateTime verifiedAt;
}
