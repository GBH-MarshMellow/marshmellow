package com.gbh.gbh_mm.user.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.gbh.gbh_mm.api.DemandDepositAPI;
import com.gbh.gbh_mm.common.exception.CustomException;
import com.gbh.gbh_mm.common.exception.ErrorCode;
import com.gbh.gbh_mm.finance.demandDeposit.vo.request.RequestFindTransactionList;
import com.gbh.gbh_mm.user.model.dto.AccountDto;
import com.gbh.gbh_mm.user.model.dto.DepositDto;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import com.gbh.gbh_mm.user.model.entity.User;
import com.gbh.gbh_mm.user.model.request.*;
import com.gbh.gbh_mm.user.model.response.*;
import com.gbh.gbh_mm.user.repo.UserRepository;
import com.gbh.gbh_mm.user.util.JwtTokenProvider;

import java.security.KeyFactory;
import java.security.PublicKey;
import java.security.spec.X509EncodedKeySpec;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.security.SecureRandom;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

@Service
@Slf4j
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;
    private final BCryptPasswordEncoder bCryptPasswordEncoder;
    private final JwtTokenProvider jwtTokenProvider;
    private final RedisTemplate<String, Object> redisTemplate;
    private final SecureRandom secureRandom = new SecureRandom();
    private final CertService certService;

    private final DemandDepositAPI demandDepositAPI;
    private final ModelMapper mapper;

    @Value("${spring.mail.username}")
    private String serverEmail;

    @Value("${rsa.pubkey}")
    private String rsaPubKey;

    public IdentityVerificationResponseDto verify(
        IdentityVerificationRequestDto identityVerificationRequestDto) {
        String phoneNumber = identityVerificationRequestDto.getPhoneNumber();
        String verificationCode = generateVerificationCode();
        int ttlTimes = 3;
        IdentityVerificationResponseDto identityVerificationResponseDto = IdentityVerificationResponseDto.
            builder()
            .serverEmail(serverEmail)
            .code(verificationCode)
            .verified(false)
            .expiresIn(ttlTimes * 60)
            .build();
        redisTemplate.opsForValue()
            .set(phoneNumber, identityVerificationResponseDto, Duration.ofMinutes(ttlTimes));
        log.info("Redis에 인증 코드 저장: key={}, value={}, 만료시간=3분", phoneNumber,
            identityVerificationResponseDto);
        return identityVerificationResponseDto;
    }

    private String generateVerificationCode() {
        int code = secureRandom.nextInt(900000) + 100000; // 6자리 인증번호 생성 (100000 ~ 999999)
        return String.valueOf(code);
    }

    @Transactional
    public SignUpResponseDto register(SignUpRequestDto signUpRequestDto) {

        // 중복회원 검증을 핸드폰 번호로 해버리면~~~? -> 번호이동 -> 새로 회원가입해야함?
        if (userRepository.existsByPhoneNumber(signUpRequestDto.getPhoneNumber())) {
            throw new CustomException(ErrorCode.DUPLICATE_RESOURCE);
        }
        String userCode = signUpRequestDto.getUserCode();
        if (Objects.isNull(userCode) || !userCode.matches("\\d{6}-[1-4]")) {
            throw new CustomException(ErrorCode.VALIDATION_FAILED);
        }

        String birthDate = userCode.substring(0, 6);
        char gender = userCode.charAt(7);
        if (gender == '1' || gender == '3') {
            gender = 'M';
        } else if (gender == '2' || gender == '4') {
            gender = 'F';
        }

        CIResponseDto connectionInformation = certService.createConnectionInformation(
            CIRequestDto.builder()
                .userName(signUpRequestDto.getUserName())
                .phoneNumber(signUpRequestDto.getPhoneNumber())
                .userCode(signUpRequestDto.getUserCode())
                .build()
        );

        User user = User.builder()
            .userName(signUpRequestDto.getUserName())
            .phoneNumber(signUpRequestDto.getPhoneNumber())
            .birth(birthDate)
            .gender(gender)
            .connectionInformation(connectionInformation.getConnectionInformation())
            .pin(bCryptPasswordEncoder.encode(signUpRequestDto.getPin()))
            .fcmToken(signUpRequestDto.getFcmToken())
            .build();
        userRepository.save(user);
        String accessToken = jwtTokenProvider.createAccessToken(user.getUserPk());
        String refreshToken = jwtTokenProvider.createRefreshToken(user.getUserPk());

        return SignUpResponseDto.builder()
            .accessToken(accessToken)
            .refreshToken(refreshToken)
            .build();
    }

    public LoginResponseDto loginByPin(LoginByPinRequestDto loginByPinRequestDto) {

        // 회원 조회
        User user = userRepository.findByPhoneNumber(loginByPinRequestDto.getPhoneNumber())
            .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

        // pin 검증
        if (!bCryptPasswordEncoder.matches(loginByPinRequestDto.getPin(), user.getPin())) {
            throw new CustomException(ErrorCode.USER_INVALID_PIN);
        }

        String accessToken = jwtTokenProvider.createAccessToken(user.getUserPk());
        String refreshToken = jwtTokenProvider.createRefreshToken(user.getUserPk());
        return LoginResponseDto.builder()
            .accessToken(accessToken)
            .refreshToken(refreshToken)
            .build();
    }

    public LoginResponseDto loginByBio(LoginByBioRequestDto loginByBioRequestDto) {
        // 회원 조회
        User user = userRepository.findByPhoneNumber(loginByBioRequestDto.getPhoneNumber())
            .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));
        String accessToken = jwtTokenProvider.createAccessToken(user.getUserPk());
        String refreshToken = jwtTokenProvider.createRefreshToken(user.getUserPk());
        return LoginResponseDto.builder()
            .accessToken(accessToken)
            .refreshToken(refreshToken)
            .build();
    }

    public LoginResponseDto reissueTokens(String refreshToken) {
        // (1) refreshToken 유효성 검사
        if (!jwtTokenProvider.isRefreshTokenValid(refreshToken)) {
            throw new CustomException(ErrorCode.INVALID_TOKEN);
        }
        // (2) userPk 추출
        String userPk = jwtTokenProvider.getUserPk(refreshToken);

        // (3) 새 토큰 발급
        String newAccessToken = jwtTokenProvider.createAccessToken(Long.valueOf(userPk));
        String newRefreshToken = jwtTokenProvider.createRefreshToken(Long.valueOf(userPk));

        // (4) 응답
        return LoginResponseDto.builder()
            .accessToken(newAccessToken)
            .refreshToken(newRefreshToken)
            .build();
    }

    // == 로그아웃(Access Token 블랙리스트 등록, Refresh Token 삭제) ==
    public void logout(String bearerToken) {

        String accessToken = "";

        if (Objects.nonNull(bearerToken) && bearerToken.startsWith("Bearer ")) {
            accessToken = bearerToken.substring(7);
        }
        // 만료까지 남은 시간 계산
        long expiration = getRemainingExpiration(accessToken);

        // Access Token 블랙리스트 등록
        if (expiration > 0) {
            redisTemplate.opsForValue()
                .set("BL:" + accessToken, "true", expiration, TimeUnit.MILLISECONDS);
        }
        // Refresh Token 삭제
        String userPk = jwtTokenProvider.getUserPk(accessToken);
        redisTemplate.delete("RT:" + userPk);
    }

    /**
     * 안전하게 AT 남은 만료 시간을 구하는 헬퍼 (파싱 예외 방지)
     */
    private long getRemainingExpiration(String accessToken) {
        try {
            Date expiration = jwtTokenProvider.extractExpiration(accessToken);
            long now = System.currentTimeMillis();
            return expiration.getTime() - now;
        } catch (Exception e) {
            // 파싱 불가 또는 이미 만료된 경우
            return 0;
        }
    }

    public Boolean isIntegratedAuthenticated(Long userPk) {
        User user = userRepository.findByUserPk(userPk)
            .orElseThrow(() -> new RuntimeException("사용자를 찾을 수 없습니다."));
        return !Objects.isNull(user.getUserKey());
    }

    public CertResponseDto issueCertificate(ClientCertIssueRequestDto clientCertIssueRequestDto,
        Long userPk) {
        User user = userRepository.findByUserPk(userPk)
            .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

        CertIssueRequestDto certIssueRequestDto = CertIssueRequestDto.builder()
            .csrPem(clientCertIssueRequestDto.getCsrPem())
            .userEmail(clientCertIssueRequestDto.getUserEmail())
            .connectionInformation(user.getConnectionInformation())
            .userName(user.getUserName())
            .phoneNumber(user.getPhoneNumber())
            .birth(user.getBirth())
            .build();
        return certService.createCertificate(certIssueRequestDto);
    }

    public CertExistResponseDto checkCertificateExistence(Long userPk) {
        User user = userRepository.findByUserPk(userPk)
            .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

        return certService.checkCertificateExistence(CertExistRequestDto.builder()
            .connectionInformation(user.getConnectionInformation())
            .build()
        );
    }

    public DigitalSignatureIssueResponseDto issueDigitalSignature(ClientDigitalSignatureRequestDto clientDigitalSignatureRequestDto, Long userPk) {

        User user = userRepository.findByUserPk(userPk)
                .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

        DigitalSignatureIssueResponseDto dto =  certService.issueDigitalSignature(DigitalSignatureIssueRequestDto.builder()
                .signedData(clientDigitalSignatureRequestDto.getSignedData())
                .originalText(clientDigitalSignatureRequestDto.getOriginalText())
                .halfUserKey(clientDigitalSignatureRequestDto.getHalfUserKey())
                .certificatePem(clientDigitalSignatureRequestDto.getCertificatePem())
                .connectionInformation(user.getConnectionInformation())
                .orgList(clientDigitalSignatureRequestDto.getOrgList())
                .build()
        );
        user.saveUserKey(dto.getUserKey());
        userRepository.save(user);
        return dto;
    }

    public ResponseFindAccountList findAccountList(CustomUserDetails userDetails) {
        try {
            Map<String, Object> responseData = demandDepositAPI
                .findDemandDepositAccountList(userDetails.getUserKey());
            Map<String, Object> apiData = (Map<String, Object>) responseData.get("apiResponse");
            List<Map<String, Object>> recData = (List<Map<String, Object>>) apiData.get("REC");

            List<AccountDto> accountDtoList = new ArrayList<>();
            for (Map<String, Object> recDatum : recData) {
                AccountDto accountDto = mapper.map(recDatum, AccountDto.class);

                accountDtoList.add(accountDto);
            }

            ResponseFindAccountList response = ResponseFindAccountList.builder()
                .accountList(accountDtoList)
                .build();

            return response;
        } catch (CustomException e) {
            System.out.println(e.getMessage());
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }

        return null;
    }

    public ResponseDepositList findDepositList(RequestDepositList request,
        CustomUserDetails userDetails) {
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMdd");

        // 현재 날짜를 구하고 포맷팅
        LocalDate currentDate = LocalDate.now();
        String currentDateStr = currentDate.format(formatter);

        // 3개월 전 날짜를 구하고 포맷팅
        LocalDate threeMonthsAgo = currentDate.minusMonths(3);
        String threeMonthsAgoStr = threeMonthsAgo.format(formatter);

        try {
            RequestFindTransactionList apiRequest = RequestFindTransactionList.builder()
                .startDate(threeMonthsAgoStr)
                .endDate(currentDateStr)
                .userKey(userDetails.getUserKey())
                .accountNo(request.getAccountNo())
                .transactionType("M")
                .orderByType("DESC")
                .build();
            Map<String, Object> dataResponse = demandDepositAPI.findTransactionList(apiRequest);
            Map<String, Object> apiResponse = (Map<String, Object>) dataResponse.get("apiResponse");
            Map<String, Object> recData = (Map<String, Object>) apiResponse.get("REC");
            List<Map<String, Object>> transactionList =
                (List<Map<String, Object>>) recData.get("list");

            List<DepositDto> depositDtoList = new ArrayList<>();
            for (Map<String, Object> stringObjectMap : transactionList) {
                long transactionBalance =
                    Long.parseLong((String) stringObjectMap.get("transactionBalance"));
                String transactionSummary = (String) stringObjectMap.get("transactionSummary");
                String[] arr = transactionSummary.split(" ");

                if (transactionBalance >= 800000 && !arr[0].equals("(대출)")
                && !arr[0].equals("(예금)") && !arr[0].equals("예금") && !arr[0].equals("MarshMellow")) {
                    DepositDto depositDto = DepositDto.builder()
                        .transactionDate((String) stringObjectMap.get("transactionDate"))
                        .transactionTime((String) stringObjectMap.get("transactionTime"))
                        .transactionBalance(transactionBalance)
                        .transactionMemo((String) stringObjectMap.get("transactionMemo"))
                        .transactionSummary(transactionSummary)
                        .build();

                    depositDtoList.add(depositDto);
                }
            }

            ResponseDepositList response = ResponseDepositList.builder()
                .depositList(depositDtoList)
                .build();

            return response;
        } catch (CustomException e) {

        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }

        return null;
    }

    public ResponseCreateSalary createSalary(RequestCreateSalary request,
        CustomUserDetails userDetails) {
        try {
            User user = userRepository.findById(userDetails.getUserPk())
                .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

            user.setSalaryAmount(request.getSalary());
            user.setSalaryDate(request.getDate());
            user.setSalaryAccount(request.getAccount());

            userRepository.save(user);

            ResponseCreateSalary response = ResponseCreateSalary.builder()
                .message("SUCCESS")
                .build();

            return response;
        } catch (CustomException e) {
            System.out.println(e.getMessage());
        }

        ResponseCreateSalary response = ResponseCreateSalary.builder()
            .message("FAIL")
            .build();

        return response;
    }

    public ResponseUpdateSalary updateSalary(RequestUpdateSalary request,
        CustomUserDetails userDetails) {
        try {
            User user = userRepository.findById(userDetails.getUserPk())
                .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

            user.setSalaryAmount(request.getSalary());
            user.setSalaryDate(request.getDate());
            user.setSalaryAccount(request.getAccount());

            userRepository.save(user);

            ResponseUpdateSalary response = ResponseUpdateSalary.builder()
                .message("SUCCESS")
                .build();

            return response;
        } catch (CustomException e) {
            System.out.println(e.getMessage());
        }

        ResponseUpdateSalary response = ResponseUpdateSalary.builder()
            .message("FAIL")
            .build();

        return response;
    }

    public ResponseUserDetail findUserDetail(CustomUserDetails userDetails) {
        User user = userRepository.findByUserPk(userDetails.getUserPk())
            .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

        ResponseUserDetail response= mapper.map(user, ResponseUserDetail.class);

        if (user.getUserKey() != null) {
            response.setUserKeyYn("Y");
        } else {
            response.setUserKeyYn("N");
        }

        return response;

    }

    public Integer findSalary(CustomUserDetails userDetails) {
        User user = userRepository.findByUserPk(userDetails.getUserPk())
            .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

        return user.getSalaryDate();
    }

    @Transactional
    public String createAesKey(CustomUserDetails userDetails) {
        User user = userRepository.findByUserPk(userDetails.getUserPk())
                .orElseThrow(() -> new CustomException(ErrorCode.USER_NOT_FOUND));

        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(128);

            SecretKey aesKey = keyGen.generateKey();

            String encodedAes = Base64.getEncoder().encodeToString(aesKey.getEncoded());

            user.setAesKey(encodedAes);
            userRepository.save(user);

            byte[] rsaPubKeyByte = Base64.getDecoder().decode(rsaPubKey);
            X509EncodedKeySpec spec = new X509EncodedKeySpec(rsaPubKeyByte);
            KeyFactory keyFactory = KeyFactory.getInstance("RSA");

            PublicKey rsaPublicKey = keyFactory.generatePublic(spec);

            Cipher ciper = Cipher.getInstance("RSA/ECB/PKCS1Padding");
            ciper.init(Cipher.ENCRYPT_MODE, rsaPublicKey);
            byte[] encryptedBytes = ciper.doFinal(aesKey.getEncoded());

            return Base64.getEncoder().encodeToString(encryptedBytes);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
