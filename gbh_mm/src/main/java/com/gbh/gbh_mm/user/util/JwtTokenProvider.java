package com.gbh.gbh_mm.user.util;

import com.gbh.gbh_mm.common.exception.CustomException;
import com.gbh.gbh_mm.common.exception.ErrorCode;
import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

@Component
@Slf4j
@RequiredArgsConstructor
public class JwtTokenProvider {

    @Value("${jwt.salt}")
    private String salt;

    private SecretKey jwtKey;

    @Value("${jwt.access-token-expiration}")
    private long accessTokenExpiration;

    @Value("${jwt.refresh-token-expiration}")
    private long refreshTokenExpiration;

    @PostConstruct
    public void init() {
        this.jwtKey = Keys.hmacShaKeyFor(salt.getBytes(StandardCharsets.UTF_8));
    }

    private final RedisTemplate<String, String> redisTemplate;

    public String createAccessToken(Long userPk) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("userPk", userPk);
        claims.put("tokenType", "ACCESS");
        return generateToken(claims, "access-token", accessTokenExpiration);
    }

    public String createRefreshToken(Long userPk) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("userPk", userPk);
        claims.put("tokenType", "REFRESH");
        String refreshToken = generateToken(claims, "refresh-token", refreshTokenExpiration);
        redisTemplate.opsForValue().set("RT:" + userPk, refreshToken, refreshTokenExpiration, TimeUnit.MILLISECONDS);
        return refreshToken;
    }

    private String generateToken(Map<String, Object> claims, String subject, long expireTime) {

        return Jwts.builder().header().add(Map.of("typ","JWT")).and().claims(claims).subject(subject)
                .issuedAt(new Date(System.currentTimeMillis()))
                .expiration(new Date(System.currentTimeMillis() + expireTime)).signWith(jwtKey).compact();
    }

    public boolean isAccessTokenValid(String token, UserDetails userDetails) {
        // 1) 서명/만료 체크
        if (!checkToken(token)) {
            return false;
        }
        // 2) 블랙리스트 체크
        if (Boolean.TRUE.equals(redisTemplate.hasKey("BL:" + token))) {
            return false;
        }
        // 3) 유저 ID 매칭 및 만료 확인
        final String userPk = getUserPk(token);
        final boolean isUserNameMatched = userPk.equals(userDetails.getUsername());
        return isUserNameMatched && !isTokenExpired(token);
    }

    public boolean isRefreshTokenValid(String token) {
        // 1) 서명/만료 체크
        if (!checkToken(token)) {
            return false;
        }
        // 2) Redis 저장값과 비교
        String userPk = getUserPk(token);
        String refreshTokenInRedis = redisTemplate.opsForValue().get("RT:" + userPk);

        // 3) Redis에 없거나 다르면 무효
        return token.equals(refreshTokenInRedis);
    }
    public boolean checkToken(String token) {
        try {
            Jws<Claims> claims = Jwts.parser().verifyWith(jwtKey).build().parseSignedClaims(token);
            log.debug("claims: {}", claims);
            return true;
        } catch (Exception e) {
            log.error(e.getMessage());
            return false;
        }
    }
    private boolean isTokenExpired(String token) {
        return extractExpiration(token).before(new Date());
    }

    public Date extractExpiration(String token) {
        return extractClaim(token, Claims::getExpiration);
    }

    private Claims extractAllClaims(String token) {
        return Jwts
                .parser()
                .verifyWith(jwtKey)
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }

    public <T> T extractClaim(String token, Function<Claims, T> claimsResolver) {
        final Claims claims = extractAllClaims(token);
        return claimsResolver.apply(claims);
    }

    public String getUserPk(String token) {
        Claims claims = null;
        try {
            claims = extractAllClaims(token);

        } catch (Exception e) {
            log.error(e.getMessage());
            throw new CustomException(ErrorCode.UNAUTHORIZED);
        }
        return String.valueOf(claims.get("userPk"));
    }

    /**
     * Access Token 남은 만료 시간을 ms 단위로 반환
     */
    private long getRemainingExpiration(String token) {
        Date expiration = extractExpiration(token);
        return expiration.getTime() - System.currentTimeMillis();
    }

}
