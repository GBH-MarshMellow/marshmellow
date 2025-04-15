package com.gbh.marshmellow

import android.content.Context
import android.os.Build
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import androidx.annotation.RequiresApi
import java.io.ByteArrayOutputStream
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.PrivateKey
import java.security.PublicKey
import java.security.Signature
import java.security.cert.Certificate
import java.util.Base64
import javax.crypto.Cipher

class SecureKeyManager(private val context: Context) {
    companion object {
        private const val ANDROID_KEYSTORE = "AndroidKeyStore"
        private const val KEY_ALIAS = "gbh_secure_key"
    }

    /**
     * RSA 키 쌍 생성 - 하드웨어 보안 모듈 사용 강제
     */
    @RequiresApi(Build.VERSION_CODES.M)
    fun generateSecureKeyPair(): Boolean {
        try {
            val keyPairGenerator = KeyPairGenerator.getInstance(
                KeyProperties.KEY_ALGORITHM_RSA, 
                ANDROID_KEYSTORE
            )
            
            val builder = KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT or
                        KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY
            )
                .setDigests(KeyProperties.DIGEST_SHA256)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_PKCS1)
                .setSignaturePaddings(KeyProperties.SIGNATURE_PADDING_RSA_PKCS1)
                .setKeySize(2048)
                // 하드웨어 보안 요소 사용 강제
                .setIsStrongBoxBacked(true) // StrongBox 사용 (가능한 경우)
                .setUserAuthenticationRequired(false) // 사용자 인증 필요 없음

            // Android P 이상에서만 작동
            // if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            //     builder.setUnrestrictedKeySize(true)
            // }
            
            keyPairGenerator.initialize(builder.build())
            keyPairGenerator.generateKeyPair()
            
            return isKeyInHardware()
        } catch (e: Exception) {
            // StrongBox가 지원되지 않는 경우 일반 하드웨어 기반 키스토어로 폴백
            try {
                val keyPairGenerator = KeyPairGenerator.getInstance(
                    KeyProperties.KEY_ALGORITHM_RSA, 
                    ANDROID_KEYSTORE
                )
                
                val builder = KeyGenParameterSpec.Builder(
                    KEY_ALIAS,
                    KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT or
                            KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY
                )
                    .setDigests(KeyProperties.DIGEST_SHA256)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_PKCS1)
                    .setSignaturePaddings(KeyProperties.SIGNATURE_PADDING_RSA_PKCS1)
                    .setKeySize(2048)
                    .setUserAuthenticationRequired(false)
                    // StrongBox 요구 없이 진행
                
                keyPairGenerator.initialize(builder.build())
                keyPairGenerator.generateKeyPair()
                
                return isKeyInHardware()
            } catch (e: Exception) {
                e.printStackTrace()
                return false
            }
        }
    }

    /**
     * 키가 하드웨어에 보관되어 있는지 확인
     */
    @RequiresApi(Build.VERSION_CODES.M)
    private fun isKeyInHardware(): Boolean {
        val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE)
        keyStore.load(null)
        val entry = keyStore.getEntry(KEY_ALIAS, null) as? KeyStore.PrivateKeyEntry
        
        if (entry?.privateKey != null) {
            val privateKey = entry.privateKey
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                val algorithm = privateKey.algorithm
                val keyFactory = java.security.KeyFactory.getInstance(algorithm, ANDROID_KEYSTORE)
                val keyInfo = keyFactory.getKeySpec(privateKey, android.security.keystore.KeyInfo::class.java)
                
                return keyInfo.isInsideSecureHardware
            }
        }
        return false
    }

    /**
     * 공개키 가져오기
     */
    fun getPublicKey(): String? {
        try {
            val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE)
            keyStore.load(null)
            
            val certificate = keyStore.getCertificate(KEY_ALIAS) ?: return null
            val publicKey = certificate.publicKey
            
            val encoded = publicKey.encoded
            return Base64.getEncoder().encodeToString(encoded)
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    /**
     * 데이터 서명
     */
    fun signData(data: ByteArray): String? {
        try {
            val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE)
            keyStore.load(null)
            
            val privateKey = keyStore.getKey(KEY_ALIAS, null) as PrivateKey
            val signature = Signature.getInstance("SHA256withRSA")
            signature.initSign(privateKey)
            signature.update(data)
            
            val signedData = signature.sign()
            return Base64.getEncoder().encodeToString(signedData)
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    /**
     * CSR 생성
     */
    fun generateCSR(commonName: String, organization: String, country: String): String? {
        try {
            // 실제 CSR 생성은 복잡하므로 여기서는 간단한 형태로만 표현
            // 실제 구현 시 Bouncy Castle 같은 라이브러리 사용 권장
            val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE)
            keyStore.load(null)
            
            val certificate = keyStore.getCertificate(KEY_ALIAS)
            val publicKey = certificate.publicKey
            
            // CSR 생성 로직 (간소화됨)
            val csrData = "CN=$commonName,O=$organization,C=$country".toByteArray()
            val signature = signData(csrData) ?: return null
            
            return "-----BEGIN CERTIFICATE REQUEST-----\n" +
                    Base64.getEncoder().encodeToString(csrData) + "\n" +
                    signature + "\n" +
                    "-----END CERTIFICATE REQUEST-----"
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    /**
     * 데이터 암호화
     */
    fun encryptData(plainText: String): String? {
        try {
            val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE)
            keyStore.load(null)
            
            val certificate = keyStore.getCertificate(KEY_ALIAS)
            val publicKey = certificate.publicKey
            
            val cipher = Cipher.getInstance("RSA/ECB/PKCS1Padding")
            cipher.init(Cipher.ENCRYPT_MODE, publicKey)
            
            val encryptedBytes = cipher.doFinal(plainText.toByteArray())
            return Base64.getEncoder().encodeToString(encryptedBytes)
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    /**
     * 데이터 복호화
     */
    fun decryptData(encryptedText: String): String? {
        try {
            val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE)
            keyStore.load(null)
            
            val privateKey = keyStore.getKey(KEY_ALIAS, null) as PrivateKey
            
            val cipher = Cipher.getInstance("RSA/ECB/PKCS1Padding")
            cipher.init(Cipher.DECRYPT_MODE, privateKey)
            
            val encryptedBytes = Base64.getDecoder().decode(encryptedText)
            val decryptedBytes = cipher.doFinal(encryptedBytes)
            
            return String(decryptedBytes)
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }
}