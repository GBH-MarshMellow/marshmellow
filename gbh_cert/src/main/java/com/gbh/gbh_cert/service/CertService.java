package com.gbh.gbh_cert.service;

import com.gbh.gbh_cert.exception.CustomException;
import com.gbh.gbh_cert.exception.ErrorCode;
import com.gbh.gbh_cert.model.dto.request.CIRequestDto;
import com.gbh.gbh_cert.model.dto.request.CertExistRequestDto;
import com.gbh.gbh_cert.model.dto.request.CertIssueRequestDto;
import com.gbh.gbh_cert.model.dto.request.DigitalSignatureIssueRequestDto;
import com.gbh.gbh_cert.model.dto.response.CIResponseDto;
import com.gbh.gbh_cert.model.dto.response.CertExistResponseDto;
import com.gbh.gbh_cert.model.dto.response.CertResponseDto;
import com.gbh.gbh_cert.model.dto.response.DigitalSignatureIssueResponseDto;
import com.gbh.gbh_cert.model.entity.*;
import com.gbh.gbh_cert.model.entity.Certificate;
import com.gbh.gbh_cert.model.repository.CertficateRepository;
import com.gbh.gbh_cert.util.CIGenerator;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.bouncycastle.asn1.ASN1ObjectIdentifier;
import org.bouncycastle.asn1.DERNull;
import org.bouncycastle.asn1.pkcs.PrivateKeyInfo;
import org.bouncycastle.asn1.x500.X500Name;
import org.bouncycastle.asn1.x509.AlgorithmIdentifier;
import org.bouncycastle.asn1.x509.DigestInfo;
import org.bouncycastle.cert.jcajce.JcaX509CertificateConverter;
import org.bouncycastle.cert.jcajce.JcaX509v3CertificateBuilder;
import org.bouncycastle.jce.provider.BouncyCastleProvider;
import org.bouncycastle.openssl.PEMParser;
import org.bouncycastle.openssl.jcajce.JcaPEMKeyConverter;
import org.bouncycastle.openssl.jcajce.JcaPEMWriter;
import org.bouncycastle.operator.ContentSigner;
import org.bouncycastle.operator.jcajce.JcaContentSignerBuilder;
import org.bouncycastle.pkcs.PKCS10CertificationRequest;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.crypto.Cipher;
import java.io.*;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.security.interfaces.RSAPublicKey;
import java.time.LocalDateTime;
import java.util.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class CertService {

    private final CIGenerator ciGenerator;
    private final UserService userService;
    private final CertficateRepository certficateRepository;
    private final SignatureRequestService signatureRequestService;
    private final OrganizationService organizationService;
    private final SignatureVerificationService signatureVerificationService;
    @Value("${cert.ca-private-key}")
    private Resource caPrivateKeyResource;

    @Transactional
    public CIResponseDto getConnectionInformation(CIRequestDto ciRequestDto) {
        String ci = ciGenerator.generateCi(ciRequestDto);
        userService.registerUserIfNotExist(ciRequestDto, ci);

        return CIResponseDto.builder()
                .connectionInformation(ci)
                .build();
    }
    @Transactional
    public CertResponseDto createCertificate(CertIssueRequestDto certIssueRequestDto) throws Exception {
        System.out.println("certIssueRequestDto = " + certIssueRequestDto);
        String base64Data = getBase64Data(certIssueRequestDto);

        byte[] decodedBytes = Base64.getDecoder().decode(base64Data);
        PKCS10CertificationRequest csr = new PKCS10CertificationRequest(decodedBytes);



        // 2. 발급자 (CA) 정보 설정
        //인증 기관(CA)의 세부 정보를 설정합니다.
        //현재 타임스탬프를 사용하여 고유한 일련번호를 생성합니다.
        //인증서 유효 기간을 현재 시간부터 1년 후로 설정합니다.
        X500Name issuer = new X500Name("CN=MM CA,O=MyCA,C=KR");
        BigInteger serial = BigInteger.valueOf(System.currentTimeMillis());
        Date notBefore = new Date();
        Calendar cal = Calendar.getInstance();
        cal.add(Calendar.YEAR, 1);
        Date notAfter = cal.getTime();

        // 3. 인증서 생성
        // X.509 v3 인증서를 생성합니다.
        //CA의 개인키로 인증서에 서명합니다.
        JcaX509v3CertificateBuilder certBuilder = new JcaX509v3CertificateBuilder(
                issuer, serial, notBefore, notAfter, csr.getSubject(), csr.getSubjectPublicKeyInfo());

        ContentSigner signer = new JcaContentSignerBuilder("SHA512withRSA")
                .build(loadCaPrivateKey());

        X509Certificate x509Certificateertificate = new JcaX509CertificateConverter()
                .setProvider("BC").getCertificate(certBuilder.build(signer));

        // CI로 유저 조회
        User user = userService.lookUpUserByCI(certIssueRequestDto.getConnectionInformation());

        // userKey 생성 및 반 저장 후 반 리턴
        String halfUserKey = userService.updateUserKey(user, certIssueRequestDto.getUserEmail());

        Certificate certificate = Certificate.builder()
                .user(user)
                .certSerial(x509Certificateertificate.getSerialNumber().longValue())
                .certData(convertCertificateToPem(x509Certificateertificate)) // 문자열로 저장
                .certStatus(Certificate.CertStatus.VALID)
                .build();

        certficateRepository.findByUser(user)
                .ifPresent(cert -> {
                    cert.setCertStatus(Certificate.CertStatus.EXPIRED);
                    certficateRepository.save(cert);
                });

        certficateRepository.save(certificate);

        // 5. PEM 인코딩 후 반환
        StringWriter writer = new StringWriter();
        JcaPEMWriter pemWriter = new JcaPEMWriter(writer);
        pemWriter.writeObject(x509Certificateertificate);
        pemWriter.close();
        return CertResponseDto.builder()
                .certificatePem(writer.toString())
                .halfUserKey(halfUserKey)
                .build();
    }

    private static String getBase64Data(CertIssueRequestDto certIssueRequestDto) {
        String rawCsrPem = certIssueRequestDto.getCsrPem();

        // 2. CSR PEM 문자열 정리: 캐리지 리턴 제거, 이스케이프된 줄바꿈 변환, 앞뒤 공백 제거
        String csrPem = rawCsrPem
                .replace("\r", "")
                .replace("\\n", "\n")
                .trim();

        // 3. 헤더/푸터 검증
        if (!csrPem.startsWith("-----BEGIN CERTIFICATE REQUEST-----") ||
                !csrPem.endsWith("-----END CERTIFICATE REQUEST-----")) {
            throw new IllegalArgumentException("Invalid CSR format");
        }

        // 4. 헤더와 푸터를 제거하여 Base64 데이터만 추출 후 모든 공백 제거
        return csrPem
                .replace("-----BEGIN CERTIFICATE REQUEST-----", "")
                .replace("-----END CERTIFICATE REQUEST-----", "")
                .replaceAll("\\s+", "").trim();
    }

    private PrivateKey loadCaPrivateKey() throws Exception {
        // 예: src/main/resources/ca/ca_private_key.pem
        // 여기선 간단하게 KeyStore, 또는 파일 기반 PEM 파싱으로 구현 가능
        try (InputStream inputStream = caPrivateKeyResource.getInputStream();
             PEMParser pemParser = new PEMParser(new InputStreamReader(inputStream))) {

            Object keyObject = pemParser.readObject();
            JcaPEMKeyConverter converter = new JcaPEMKeyConverter();

            // PrivateKeyInfo 또는 KeyPair 타입 처리
            if (keyObject instanceof PrivateKeyInfo) {
                return converter.getPrivateKey((PrivateKeyInfo) keyObject);
            } else if (keyObject instanceof KeyPair) {
                return ((KeyPair) keyObject).getPrivate();
            }

            throw new IllegalArgumentException("지정된 키 형식이 아닙니다.");
        }
    }

    // PEM 형식으로 인증서 변환
    private String convertCertificateToPem(X509Certificate certificate) {
        try (StringWriter writer = new StringWriter();
             JcaPEMWriter pemWriter = new JcaPEMWriter(writer)) {
            pemWriter.writeObject(certificate);
            pemWriter.flush();
            return writer.toString();
        } catch (IOException e) {
            throw new RuntimeException("Certificate PEM conversion failed", e);
        }
    }

    public CertExistResponseDto checkCertificateExistence(CertExistRequestDto certExistRequestDto) {
        User user = userService.lookUpUserByCI(certExistRequestDto.getConnectionInformation());

        return certficateRepository.findByUserAndCertStatus(user, Certificate.CertStatus.VALID)
                .map(cert -> CertExistResponseDto.builder()
                        .exist(true)
                        .status(cert.getCertStatus().name())
                        .certificatePem(cert.getCertData())
                        .build())
                .orElseGet(() -> CertExistResponseDto.builder()
                        .exist(false)
                        .build());
    }
    private boolean isCertificateMatching(String pem1, String pem2) {
        try {
            X509Certificate cert1 = convertToX509(pem1);
            X509Certificate cert2 = convertToX509(pem2);

            // 🔍 DER 바이트 배열로 직접 비교
            byte[] der1 = cert1.getEncoded();
            byte[] der2 = cert2.getEncoded();

            boolean isEqual = Arrays.equals(der1, der2);

            if (!isEqual) {
                log.warn("❗ 인증서 DER 비교 결과 다름!");
                log.warn("📄 서버 인증서(Base64 DER): " + Base64.getEncoder().encodeToString(der1));
                log.warn("📄 클라 인증서(Base64 DER): " + Base64.getEncoder().encodeToString(der2));
            }

            return isEqual;
        } catch (Exception e) {
            log.error("❌ 인증서 비교 중 오류 발생: " + e.getMessage(), e);
            return false;
        }
    }


    private X509Certificate convertToX509(String pem) throws Exception {
        CertificateFactory factory = CertificateFactory.getInstance("X.509");
        ByteArrayInputStream stream = new ByteArrayInputStream(pem.getBytes(StandardCharsets.UTF_8));
        return (X509Certificate) factory.generateCertificate(stream);
    }
    public DigitalSignatureIssueResponseDto verifyDigitalSignature(DigitalSignatureIssueRequestDto request) throws Exception {

        User user = userService.lookUpUserByCI(request.getConnectionInformation());

        Certificate certificate = certficateRepository.findByUserAndCertStatus(user, Certificate.CertStatus.VALID)
                .orElseThrow(() -> new CustomException(ErrorCode.CHILD_NOT_FOUND));
        if (!isCertificateMatching(certificate.getCertData(), request.getCertificatePem())) {
            throw new CustomException(ErrorCode.CERTFICATE_NOT_EQUALS);
        }

        // 3. 서명 검증
        boolean verified = verifyWithDigestInfoComparison(request.getOriginalText(), request.getSignedData(), convertToX509(certificate.getCertData()));
        if (!verified) {
            return DigitalSignatureIssueResponseDto.builder()
                    .verified(false)
                    .message("전자서명 검증 실패")
                    .build();
        }

        // 4. 기관 리스트 가져오기
        List<Organization> organizations = organizationService.getValidOrganizations(request.getOrgList());

        // 5. SignatureRequest 17건 저장
        List<SignatureRequest> requestList = signatureRequestService.storeSignatureRequest(user, certificate, request, organizations);

        // 6. 각 요청마다 Verification 저장
        for (SignatureRequest req : requestList) {
            signatureVerificationService.issueRequest(req);
        }

        // 7. 응답 반환
        return DigitalSignatureIssueResponseDto.builder()
                .verified(true)
                .userKey(request.getHalfUserKey()+user.getHalfUserKey())
                .message("전자서명 검증 성공 및 트랜잭션 저장 완료")
                .build();
    }

    public boolean verifyWithDigestInfoComparison(String originalText, String signedData, X509Certificate cert) {
        try {
            PublicKey publicKey = cert.getPublicKey();
            byte[] signedBytes = Base64.getDecoder().decode(signedData);

            // 1. SHA-512 Digest 생성
            byte[] digest = MessageDigest.getInstance("SHA-512").digest(originalText.getBytes(StandardCharsets.UTF_8));

            // 2. DigestInfo ASN.1 구조 생성
            AlgorithmIdentifier algId = new AlgorithmIdentifier(
                    new ASN1ObjectIdentifier("2.16.840.1.101.3.4.2.3"), DERNull.INSTANCE); // SHA-512
            DigestInfo digestInfo = new DigestInfo(algId, digest);
            byte[] expected = digestInfo.getEncoded("DER");

            // 3. 복호화
            Cipher cipher = Cipher.getInstance("RSA/ECB/PKCS1Padding");
            cipher.init(Cipher.DECRYPT_MODE, publicKey);
            byte[] decrypted = cipher.doFinal(signedBytes);

            boolean match = Arrays.equals(decrypted, expected);

            if (!match) {
                log.warn("❌ DigestInfo mismatch!");
                log.warn("📝 서버 DigestInfo: {}", Base64.getEncoder().encodeToString(expected));
                log.warn("📥 복호화 결과     : {}", Base64.getEncoder().encodeToString(decrypted));
            }

            return match;
        } catch (Exception e) {
            log.error("전자서명 DigestInfo 비교 실패", e);
            return false;
        }
    }

}