package com.gbh.gbh_mm.config.filter;

import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

import com.gbh.gbh_mm.user.util.JwtTokenProvider;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@RequiredArgsConstructor
@Slf4j
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtTokenProvider jwtTokenProvider;
    private final UserDetailsService userDetailsService;
    private final RedisTemplate<String, Object> redisTemplate;

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String uri = request.getRequestURI();
        return uri.equals("/gmail/webhook"); // ✅ Pub/Sub webhook은 필터 제외
    }

    @Override
    protected void doFilterInternal(
            @NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response,
            @NonNull FilterChain filterChain)
            throws ServletException, IOException {
        final String JWT_PREFIX = "Bearer ";
        final String JWT_HEADER_KEY = "Authorization";
        final String authHeader = request.getHeader(JWT_HEADER_KEY);
        try {
            // Authorization 헤더 확인
            if (Objects.isNull(authHeader) || !authHeader.startsWith(JWT_PREFIX)) {
                filterChain.doFilter(request, response);
                return;
            }
            String jwt = authHeader.substring(JWT_PREFIX.length());

            if (Boolean.TRUE.equals(redisTemplate.hasKey("BL:" + jwt))) {
                log.warn("🛑 사용이 차단된 토큰입니다 (블랙리스트)");
                response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                response.getWriter().write("Token is blacklisted");
                return;
            }

            String userPk = jwtTokenProvider.getUserPk(jwt); // JWT 토큰에서 유저 ID 추출
            // 유저 ID와 인증 정보 확인
            if (Objects.nonNull(userPk) && Objects.isNull(SecurityContextHolder.getContext().getAuthentication())) {
                UserDetails userDetails = userDetailsService.loadUserByUsername(userPk);

                // 토큰 유효성 검사
                if (jwtTokenProvider.isAccessTokenValid(jwt, userDetails)) {
                    UsernamePasswordAuthenticationToken authentication = new UsernamePasswordAuthenticationToken(
                            userDetails,
                            null,
                            userDetails.getAuthorities()
                    );
                    authentication.setDetails( new WebAuthenticationDetailsSource().buildDetails(request));
                    SecurityContextHolder.getContext().setAuthentication(authentication);
                } else {
                    // 유효하지 않은 토큰에 대해 401 Unauthorized 반환
                    log.warn("Invalid JWT Token");
                    response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
                    response.getWriter().write("Token Expired");
                    return;
                }
            }
        } catch (Exception e) {
            // JWT 처리 중 예외 발생 시 401 Unauthorized 반환
            log.error("Authentication error: {}", e.getMessage());

            // 문제의 토큰을 블랙리스트에 등록
            if (authHeader != null && authHeader.startsWith(JWT_PREFIX)) {
                String jwt = authHeader.substring(JWT_PREFIX.length());
                redisTemplate.opsForValue().set("BL:" + jwt, "invalid", 1, TimeUnit.HOURS);
            }

            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            response.getWriter().write("Invalid or unsupported token");
            return;
        }

        // 정상적인 요청은 다음 필터로 전달
        filterChain.doFilter(request, response);
    }

}