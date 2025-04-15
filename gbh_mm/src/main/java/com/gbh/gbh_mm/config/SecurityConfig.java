package com.gbh.gbh_mm.config;

import com.gbh.gbh_mm.config.filter.JwtAuthenticationFilter;
import com.gbh.gbh_mm.user.repo.UserRepository;
import com.gbh.gbh_mm.user.service.CustomUserDetailService;
import com.gbh.gbh_mm.user.util.JwtTokenProvider;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import java.util.List;

@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
@Slf4j
public class SecurityConfig {

    private final CustomUserDetailService userDetailsService;
    private final JwtTokenProvider jwtTokenProvider;
    private final UserRepository userRepository;
    private final RedisTemplate<String, Object> redisTemplate;

    @Bean
    public BCryptPasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public AuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider authProvider = new DaoAuthenticationProvider();
        authProvider.setUserDetailsService(userDetailsService);
        authProvider.setPasswordEncoder(passwordEncoder());
        return authProvider;
    }

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration config)
        throws Exception {
        return config.getAuthenticationManager();
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            // CORS 설정
            // CSRF 비활성화
            .csrf(AbstractHttpConfigurer::disable)
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))
            // Form 로그인 비활성화
            // Basic 인증도 비활성화
            // 세션 사용 안 함(STATELESS)
            .formLogin(AbstractHttpConfigurer::disable)
            .httpBasic(AbstractHttpConfigurer::disable)
            .sessionManagement(
                session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .exceptionHandling(exceptions -> exceptions
                .authenticationEntryPoint((request, response, authException) -> {
                    log.warn("🔴 [401 Unauthorized] 인증되지 않은 사용자 접근 - 요청 경로: {}",
                        request.getRequestURI());
                    response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Unauthorized");
                })
                .accessDeniedHandler((request, response, accessDeniedException) -> {
                    log.warn("🟠 [403 Forbidden] 권한 부족 - 요청 경로: {}, 사용자: {}",
                        request.getRequestURI(),
                        request.getUserPrincipal() != null ? request.getUserPrincipal().getName()
                            : "Anonymous");
                    response.sendError(HttpServletResponse.SC_FORBIDDEN, "Forbidden");
                })
            )
            // 인증/인가 설정
            .authorizeHttpRequests(auth -> auth
                // Swagger UI 경로 인증 없이 허용
                .requestMatchers(
                    "/v3/api-docs/**",  // OpenAPI 문서 JSON
                    "/swagger-ui/**",   // Swagger UI 리소스
                    "/swagger-ui.html", // Swagger UI 접속 페이지
                    "/webjars/**",      // Swagger가 사용하는 정적 리소스
                    "/swagger-resources/**"
                ).permitAll()
                .requestMatchers(
                    "/api/mm/auth/identity-verify", "/api/mm/auth/sign-up",
                    "/api/mm/auth/login/**", "/api/mm/auth/subscribe/**",
                    "/api/mm/auth/webhook", "/health-check",
                    "/api/mm/auth/reissue", "/actuator/**", "/gmail/webhook",
                    "/presentation/**", "/**"
                ).permitAll()
                .anyRequest().authenticated()
            )
            // JWT 필터
            .addFilterBefore(
                new JwtAuthenticationFilter(jwtTokenProvider, userDetailsService, redisTemplate),
                UsernamePasswordAuthenticationFilter.class);
        return http.build();


    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration config = new CorsConfiguration();
        config.setAllowedOrigins(List.of("http://localhost"));
        config.setAllowedMethods(List.of("GET", "POST", "PUT", "DELETE", "OPTIONS"));
        config.setAllowedHeaders(List.of("*")); // 모든 헤더 허용
        config.setAllowCredentials(true); // 인증 정보 허용

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config); // 모든 경로에 대해 적용
        return source;
    }

}
