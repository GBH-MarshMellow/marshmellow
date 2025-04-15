package com.gbh.marshmellow

import android.content.Context
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

class SecureKeyPlugin: FlutterPlugin, MethodCallHandler {
    private lateinit var channel: MethodChannel
    private lateinit var context: Context
    private lateinit var secureKeyManager: SecureKeyManager

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, "com.your.package/secure_keys")
        context = binding.applicationContext
        secureKeyManager = SecureKeyManager(context)
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "generateKeyPair" -> {
                val success = secureKeyManager.generateSecureKeyPair()
                result.success(mapOf("success" to success))
            }
            "getPublicKey" -> {
                val publicKey = secureKeyManager.getPublicKey()
                if (publicKey != null) {
                    result.success(mapOf("publicKey" to publicKey))
                } else {
                    result.error("GET_KEY_ERROR", "Failed to get public key", null)
                }
            }
            "signData" -> {
                val data = call.argument<String>("data") ?: ""
                val signature = secureKeyManager.signData(data.toByteArray())
                if (signature != null) {
                    result.success(mapOf("signature" to signature))
                } else {
                    result.error("SIGN_ERROR", "Failed to sign data", null)
                }
            }
            "generateCSR" -> {
                val commonName = call.argument<String>("commonName") ?: ""
                val organization = call.argument<String>("organization") ?: "GBH"
                val country = call.argument<String>("country") ?: "KR"
                
                val csr = secureKeyManager.generateCSR(commonName, organization, country)
                if (csr != null) {
                    result.success(mapOf("csr" to csr))
                } else {
                    result.error("CSR_ERROR", "Failed to generate CSR", null)
                }
            }
            "encryptData" -> {
                val data = call.argument<String>("data") ?: ""
                val encrypted = secureKeyManager.encryptData(data)
                if (encrypted != null) {
                    result.success(mapOf("encryptedData" to encrypted))
                } else {
                    result.error("ENCRYPT_ERROR", "Failed to encrypt data", null)
                }
            }
            "decryptData" -> {
                val data = call.argument<String>("data") ?: ""
                val decrypted = secureKeyManager.decryptData(data)
                if (decrypted != null) {
                    result.success(mapOf("decryptedData" to decrypted))
                } else {
                    result.error("DECRYPT_ERROR", "Failed to decrypt data", null)
                }
            }
            else -> {
                result.notImplemented()
            }
        }
    }
}