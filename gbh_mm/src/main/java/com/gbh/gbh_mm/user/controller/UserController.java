package com.gbh.gbh_mm.user.controller;

import com.gbh.gbh_mm.common.dto.ApiResponse;
import com.gbh.gbh_mm.user.model.entity.CustomUserDetails;
import com.gbh.gbh_mm.user.model.request.*;
import com.gbh.gbh_mm.user.model.response.*;
import com.gbh.gbh_mm.user.service.UserService;
import io.swagger.v3.oas.annotations.Operation;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/mm/auth")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @PostMapping("/identity-verify")
    public IdentityVerificationResponseDto requestVerification(@RequestBody IdentityVerificationRequestDto identityVerificationRequestDto) {
        return userService.verify(identityVerificationRequestDto);
    }

    @PostMapping("/sign-up")
    public SignUpResponseDto signUp(@RequestBody SignUpRequestDto signUpRequestDto) {
        return userService.register(signUpRequestDto);
    }

    @PostMapping("/login/pin")
    public LoginResponseDto loginByPin(@RequestBody LoginByPinRequestDto loginByPinRequestDto) {
        return userService.loginByPin(loginByPinRequestDto);
    }

    @PostMapping("/login/bio")
    public LoginResponseDto loginByBio(@RequestBody LoginByBioRequestDto loginByBioRequestDto) {
        return userService.loginByBio(loginByBioRequestDto);
    }

    @PostMapping("/reissue")
    public LoginResponseDto reissueToken(@RequestBody ReissueTokenRequestDto reissueTokenRequestDto) {
        return userService.reissueTokens(reissueTokenRequestDto.getRefreshToken());
    }

    @PostMapping("/logout")
    public ApiResponse<?> logout(@RequestHeader("Authorization") String bearerToken) {
        userService.logout(bearerToken);
        return ApiResponse.builder().code(200).message("로그아웃 성공").build();
    }

    @GetMapping("/cert/exist")
    public CertExistResponseDto isExistCertificate(@AuthenticationPrincipal CustomUserDetails userDetails) {
        return userService.checkCertificateExistence(userDetails.getUserPk());
    }

    @PostMapping("/cert/issue")
    public CertResponseDto issueCertificate(@RequestBody ClientCertIssueRequestDto clientCertIssueRequestDto,
                                            @AuthenticationPrincipal CustomUserDetails userDetails) {
        return userService.issueCertificate(clientCertIssueRequestDto, userDetails.getUserPk());
    }

    // 통합인증 진행했는지 여부 검증(userKey가 있거나 없거나로 가야하나?)
    @GetMapping("/integrated-status")
    public Boolean checkIntegratedStatus(@AuthenticationPrincipal CustomUserDetails userDetails) {
        return userService.isIntegratedAuthenticated(userDetails.getUserPk());
    }

    @PostMapping("/digital-signature")
    public DigitalSignatureIssueResponseDto issueDigitalSignature(@RequestBody ClientDigitalSignatureRequestDto digitalSignatureIssueRequestDto,
                                                                  @AuthenticationPrincipal CustomUserDetails userDetails) {
        return userService.issueDigitalSignature(digitalSignatureIssueRequestDto, userDetails.getUserPk());
    }
    @GetMapping("/account-list")
    public ResponseFindAccountList findAccountList(
        @AuthenticationPrincipal CustomUserDetails userDetails) {
        return userService.findAccountList(userDetails);
    }

    @GetMapping("/deposit-list")
    public ResponseDepositList findDepositList(
        @RequestBody RequestDepositList request,
        @AuthenticationPrincipal CustomUserDetails userDetails
    ) {
        return userService.findDepositList(request, userDetails);
    }

    @PostMapping("/salary")
    public ResponseCreateSalary createSalary(
        @RequestBody RequestCreateSalary request,
        @AuthenticationPrincipal CustomUserDetails userDetails
    ) {
        return userService.createSalary(request, userDetails);
    }

    @PatchMapping("/salary")
    public ResponseUpdateSalary updateSalary(
        @RequestBody RequestUpdateSalary request,
        @AuthenticationPrincipal CustomUserDetails userDetails
    ) {
        return userService.updateSalary(request, userDetails);
    }

    @GetMapping("/detail")
    public ResponseUserDetail findUserDetail
        (@AuthenticationPrincipal CustomUserDetails userDetails) {
        return userService.findUserDetail(userDetails);
    }

    @GetMapping("/salary")
    public Integer findsalary(@AuthenticationPrincipal CustomUserDetails userDetails) {
        return userService.findSalary(userDetails);
    }

    @PostMapping("/key-gen")
    public String createKey(@AuthenticationPrincipal CustomUserDetails userDetails) {
        return userService.createAesKey(userDetails);
    }


}
