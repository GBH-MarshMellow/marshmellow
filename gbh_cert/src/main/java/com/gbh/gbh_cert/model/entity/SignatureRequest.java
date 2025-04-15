package com.gbh.gbh_cert.model.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "tb_signature_request")
@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SignatureRequest {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "signature_req_id", nullable = false)
    private Long signatureReqId;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne
    @JoinColumn(name = "org_code", referencedColumnName = "org_code", nullable = false)
    private Organization organization;

    @ManyToOne
    @JoinColumn(name = "cert_id", nullable = false)
    private Certificate certificate;

    @Column(name = "nonce")
    private String nonce;

    @Column(name = "original_text", columnDefinition = "TEXT")
    private String originalText;

    @Column(name = "requested_at")
    private LocalDateTime requestedAt;

    @Column(name = "signature_data", columnDefinition = "TEXT")
    private String signatureData;

}
