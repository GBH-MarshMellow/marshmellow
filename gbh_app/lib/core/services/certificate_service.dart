import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:asn1lib/asn1lib.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:marshmellow/core/constants/storage_keys.dart';
import 'package:pointycastle/api.dart';
import 'package:pointycastle/asymmetric/api.dart';
import 'package:pointycastle/export.dart';
import 'package:pointycastle/key_generators/api.dart';
import 'package:pointycastle/key_generators/rsa_key_generator.dart';
import 'package:pointycastle/random/fortuna_random.dart';
import 'package:pointycastle/signers/rsa_signer.dart';
import 'package:pointycastle/digests/sha512.dart';
import 'package:crypto/crypto.dart';
/*
 TEE/SE 하드웨어 접근 불가능시 사용할
 공개키 / 개인키
*/
class CertificateService {
  final FlutterSecureStorage _secureStorage;
  static const String _privateKeyKey = 'private_key'; // 개인키를 열기 위한 키
  static const String _publicKeyKey = 'public_key'; // 공개키를 열기 위한 키

  CertificateService(this._secureStorage);

  // RSA 키 페어 생성 (pointycastle 패키지 사용)
  Future<AsymmetricKeyPair<RSAPublicKey, RSAPrivateKey>> generateRSAKeyPair({int bitLength = 2048}) async {
    try {
        // 보안 랜덤 생성기
        final secureRandom = FortunaRandom();
        final random = Random.secure();
        final seeds = List<int>.generate(32, (_) => random.nextInt(256));
        secureRandom.seed(KeyParameter(Uint8List.fromList(seeds)));

        // RSA 키 생성기
        final keyGen = RSAKeyGenerator();
        
        // 키 생성 파라미터 
        final keyParams = RSAKeyGeneratorParameters(
            BigInt.from(65537),  // publicExponent
            2048,                // keySize
            64                   // certainty
        );
        
        // 랜덤 파라미터와 함께 초기화
        keyGen.init(ParametersWithRandom(keyParams, secureRandom));

        // 키 페어 생성
        final keyPair = keyGen.generateKeyPair();
        print('어디까지 가능?');
        // 명시적 타입 체크 및 변환
        final publicKey = keyPair.publicKey as RSAPublicKey;
        final privateKey = keyPair.privateKey as RSAPrivateKey;
        print('여기 안되지?');
        return AsymmetricKeyPair(publicKey, privateKey);
    } catch (e, stackTrace) {
        print('키 페어 생성 중 오류 발생: $e');
        print('스택 트레이스: $stackTrace');
        rethrow;
    }
}

  // 키 페어 저장
  Future<void> storeKeyPair(AsymmetricKeyPair<RSAPublicKey, RSAPrivateKey> keyPair) async {
    try {
      print("storeKeyPair 실행됨!");
      // 개인키를 PEM 형식으로 인코딩하여 TEE/SE에 저장
      final privateKeyPem = _encodeRSAPrivateKeyToPem(keyPair.privateKey);
      print("privateKey 변환 성공!");
      await _secureStorage.write(key: _privateKeyKey, value: privateKeyPem);
      print("privateKey 저장 성공!");

      // 공개키도 저장
      final publicKeyPem = _encodeRSAPublicKeyToPem(keyPair.publicKey);
      print("publicKey 변환 성공!");
      await _secureStorage.write(key: _publicKeyKey, value: publicKeyPem);
      print("publicKey 저장 성공!");
    } catch (e, stacktrace) {
      print("storeKeyPair 실패: $e");
      print(stacktrace);
    }
  }

  // 개인키를 PEM 형식으로 인코딩
  String _encodeRSAPrivateKeyToPem(RSAPrivateKey privateKey) {
    // ASN.1 형식으로 인코딩
    final privateKeySequence = ASN1Sequence();
    privateKeySequence.add(ASN1Integer(BigInt.from(0))); // 버전
    privateKeySequence.add(ASN1Integer(privateKey.n!)); // Modulus
    privateKeySequence.add(ASN1Integer(privateKey.publicExponent!)); // Public Exponent
    privateKeySequence.add(ASN1Integer(privateKey.privateExponent!)); // Private Exponent
    privateKeySequence.add(ASN1Integer(privateKey.p!)); // Prime 1
    privateKeySequence.add(ASN1Integer(privateKey.q!)); // Prime 2
    privateKeySequence.add(ASN1Integer(privateKey.privateExponent! % (privateKey.p! - BigInt.from(1)))); // Exponent 1
    privateKeySequence.add(ASN1Integer(privateKey.privateExponent! % (privateKey.q! - BigInt.from(1)))); // Exponent 2
    privateKeySequence.add(ASN1Integer(privateKey.q!.modInverse(privateKey.p!))); // Coefficient

    final bytes = privateKeySequence.encodedBytes;
    final base64PrivateKey = base64.encode(bytes);
    
    return '-----BEGIN RSA PRIVATE KEY-----\n' +
        base64PrivateKey.replaceAllMapped(RegExp('.{64}'), (match) => '${match.group(0)}\n') +
        (base64PrivateKey.length % 64 == 0 ? '' : '\n') +
        '-----END RSA PRIVATE KEY-----';
  }

  // 공개키를 PEM 형식으로 인코딩
  String _encodeRSAPublicKeyToPem(RSAPublicKey publicKey) {
    final publicKeySequence = _encodePublicKey(publicKey);
    final bytes = publicKeySequence.encodedBytes;
    final base64PublicKey = base64.encode(bytes);
    
    return '-----BEGIN PUBLIC KEY-----\n' +
        base64PublicKey.replaceAllMapped(RegExp('.{64}'), (match) => '${match.group(0)}\n') +
        (base64PublicKey.length % 64 == 0 ? '' : '\n') +
        '-----END PUBLIC KEY-----';
  }

  // 저장된 개인키 가져오기
  Future<RSAPrivateKey?> getPrivateKey() async {
    final privateKeyPem = await _secureStorage.read(key: _privateKeyKey);
    if (privateKeyPem == null) return null;
    final key = _decodeRSAPrivateKeyFromPem(privateKeyPem);
    print("🔐 클라 개인키 modulus: ${key.n}");
    print("🔐 클라 개인키 exponent (private): ${key.d}");
    return key;
  }

  // 저장된 공개키 가져오기
  Future<RSAPublicKey?> getPublicKey() async {
    final publicKeyPem = await _secureStorage.read(key: _publicKeyKey);
    if (publicKeyPem == null) return null;

      final key = _decodeRSAPublicKeyFromPem(publicKeyPem);
    print("🔐 클라 공개키 modulus: ${key.modulus}");
    print("🔐 클라 공개키 exponent: ${key.exponent}");
    return key;
  }

  // PEM 형식의 개인키 디코딩
  RSAPrivateKey _decodeRSAPrivateKeyFromPem(String pemString) {
    // PEM에서 base64 추출
    final pemContent = pemString
        .replaceAll('-----BEGIN RSA PRIVATE KEY-----', '')
        .replaceAll('-----END RSA PRIVATE KEY-----', '')
        .replaceAll('\n', '');
    
    final bytes = base64.decode(pemContent);
    final asn1Parser = ASN1Parser(Uint8List.fromList(bytes));
    final topLevelSeq = asn1Parser.nextObject() as ASN1Sequence;
    
    // ASN.1 구조에서 개인키 파라미터 추출
    // 버전은 무시 (index 0)
    final modulus = (topLevelSeq.elements[1] as ASN1Integer).valueAsBigInteger;
    final publicExponent = (topLevelSeq.elements[2] as ASN1Integer).valueAsBigInteger;
    final privateExponent = (topLevelSeq.elements[3] as ASN1Integer).valueAsBigInteger;
    final p = (topLevelSeq.elements[4] as ASN1Integer).valueAsBigInteger;
    final q = (topLevelSeq.elements[5] as ASN1Integer).valueAsBigInteger;
    
    return RSAPrivateKey(modulus!, privateExponent!, p, q);
  }

  // PEM 형식의 공개키 디코딩
  RSAPublicKey _decodeRSAPublicKeyFromPem(String pemString) {
    // PEM에서 base64 추출
    final pemContent = pemString
        .replaceAll('-----BEGIN PUBLIC KEY-----', '')
        .replaceAll('-----END PUBLIC KEY-----', '')
        .replaceAll('\n', '');
    
    final bytes = base64.decode(pemContent);
    final asn1Parser = ASN1Parser(Uint8List.fromList(bytes));
    final topLevelSeq = asn1Parser.nextObject() as ASN1Sequence;
    
    final bitString = topLevelSeq.elements[1] as ASN1BitString;
    final publicKeyParser = ASN1Parser(bitString.contentBytes()!);
    final publicKeySeq = publicKeyParser.nextObject() as ASN1Sequence;
    
    final modulus = (publicKeySeq.elements[0] as ASN1Integer).valueAsBigInteger;
    final exponent = (publicKeySeq.elements[1] as ASN1Integer).valueAsBigInteger;
    
    return RSAPublicKey(modulus!, exponent!);
  }

  // 키 페어 존재 여부 확인
  Future<bool> hasKeyPair() async {
    final publicKey = await getPublicKey();
    return publicKey != null;
  }

// ✅ 수정된 generateCSR 함수: BouncyCastle 호환 Subject 구조
Future<String> generateCSR({
  required String commonName,
  String country = 'KR',
  String organization = 'GBH',
}) async {
  // 저장된 키 페어 불러오기
  print("1. 키 불러오기 시도");
  final privateKey = await getPrivateKey();
  final publicKey = await getPublicKey();
  print("2. 키 불러오기 성공");

  if (privateKey == null || publicKey == null) {
    throw Exception('키 페어가 존재하지 않습니다. 먼저 키 페어를 생성하세요.');
  }

  // 1. Subject 정보 구성 (PKCS#10 Name 구조)
  print("3. Subject 만들기 시작");
final attributes = [
  {
    'oid': ASN1ObjectIdentifier.fromComponents([2, 5, 4, 6]),  // C (2.5.4.6)
    'value': ASN1PrintableString(country)
  },
  {
    'oid': ASN1ObjectIdentifier.fromComponents([2, 5, 4, 10]), // O (2.5.4.10)
    'value': ASN1PrintableString(organization)
  },
  {
    'oid': ASN1ObjectIdentifier.fromComponents([2, 5, 4, 3]),  // CN (2.5.4.3)
    'value': ASN1PrintableString(commonName)
  },
];

final subjectSequence = ASN1Sequence();
for (var attr in attributes) {
  final attrSeq = ASN1Sequence();
  final attributeType = attr['oid'] as ASN1ObjectIdentifier;

  attrSeq.add(attributeType);
  attrSeq.add(attr['value'] as ASN1Object);

  final rdnSet = ASN1Set();
  rdnSet.add(attrSeq);
  subjectSequence.add(rdnSet);
}
print("4. Subject 구성 완료");
print("5. 공개키 인코딩 시작");
  // 2. 공개키 정보 (SubjectPublicKeyInfo)
  final publicKeyInfo = _encodePublicKey(publicKey);
  print("6. 공개키 인코딩 완료");
  // 3. CertificationRequestInfo 구성
  final csrInfoSeq = ASN1Sequence()
    ..add(ASN1Integer(BigInt.from(0))) // 버전 (v1)
    ..add(subjectSequence) // Subject
    ..add(publicKeyInfo); // SubjectPublicKeyInfo

  // 4. CSR 정보 바이트 인코딩
  final csrInfoBytes = csrInfoSeq.encodedBytes;

  // 5. 서명 생성 (SHA-512 with RSA)
  final signer = RSASigner(SHA512Digest(), '2a864886f70d01010d'); // OID: 1.2.840.113549.1.1.13 (sha512WithRSAEncryption)
  signer.init(true, PrivateKeyParameter<RSAPrivateKey>(privateKey));
  final signature = signer.generateSignature(Uint8List.fromList(csrInfoBytes)) as RSASignature;
  final signatureBytes = signature.bytes;

  // 6. 서명 알고리즘 식별자
  final algorithmSeq = ASN1Sequence()
    ..add(ASN1ObjectIdentifier(Uint8List.fromList([42, 134, 72, 134, 247, 13, 1, 1, 13]))) // 1.2.840.113549.1.1.13
    ..add(ASN1Null());

  // 7. 최종 CSR 구성 (CertificationRequest)
  final csrSequence = ASN1Sequence()
    ..add(csrInfoSeq) // CertificationRequestInfo
    ..add(algorithmSeq) // SignatureAlgorithm
    ..add(ASN1BitString(Uint8List.fromList(signatureBytes), unusedbits: 0)); // Signature (unusedBits=0 명시)

  // 8. PEM 형식으로 인코딩
  final csrBytes = csrSequence.encodedBytes;
  final csrBase64 = base64.encode(csrBytes);
  final csrPem = '-----BEGIN CERTIFICATE REQUEST-----\n' +
      csrBase64.replaceAllMapped(RegExp('.{64}'), (match) => '${match.group(0)}\n') +
      (csrBase64.length % 64 == 0 ? '' : '\n') +
      '-----END CERTIFICATE REQUEST-----';

  return csrPem;
}

// 공개키를 SubjectPublicKeyInfo로 인코딩
ASN1Sequence _encodePublicKey(RSAPublicKey publicKey) {
    final subjectPublicKeyInfo = ASN1Sequence();
    
    // Algorithm identifier
    final algorithm = ASN1Sequence();
    algorithm.add(ASN1ObjectIdentifier.fromComponents([1, 2, 840, 113549, 1, 1, 1])); // rsaEncryption
    algorithm.add(ASN1Null());
    subjectPublicKeyInfo.add(algorithm);
    
    // Public key data
    final publicKeyASN1 = ASN1Sequence();
    publicKeyASN1.add(ASN1Integer(publicKey.modulus!));
    publicKeyASN1.add(ASN1Integer(publicKey.exponent!));
    
    final publicKeyDER = publicKeyASN1.encodedBytes;
    subjectPublicKeyInfo.add(ASN1BitString(publicKeyDER));
    
    return subjectPublicKeyInfo;
  }

  // OID 매핑 함수 (String이 아닌 List<int> 반환)
  List<int> _getOIDForName(String name) {
    // OID 매핑
    final Map<String, List<int>> oidMap = {
      'C': [85, 4, 6],     // 2.5.4.6 (국가)
      'O': [85, 4, 10],    // 2.5.4.10 (조직명)
      'OU': [85, 4, 11],   // 2.5.4.11 (조직 단위)
      'CN': [85, 4, 3],    // 2.5.4.3 (공통명)
    };
    
    return oidMap[name] ?? [85, 4, 3]; // 기본값은 CN
  }

  // 전자서명(SHA-512 + RSA)
Future<String?> signData(String originalText) async {
  try {
    final privateKeyPem = await _secureStorage.read(key: StorageKeys.privateKey);
    if (privateKeyPem == null) throw Exception('개인 키를 찾을 수 없습니다.');
    final privateKey = _parsePrivateKeyFromPem(privateKeyPem);
    final normalizedText = originalText;

    // 1. SHA-512 해시
    final digest = SHA512Digest().process(utf8.encode(normalizedText) as Uint8List);

    // 2. ASN.1 DigestInfo 생성
    final digestInfo = ASN1Sequence()
      ..add(ASN1Sequence() // AlgorithmIdentifier
        ..add(ASN1ObjectIdentifier.fromComponents([2, 16, 840, 1, 101, 3, 4, 2, 3])) // OID: 2.16.840.1.101.3.4.2.3
        ..add(ASN1Null()))
      ..add(ASN1OctetString(digest));

    final digestInfoBytes = digestInfo.encodedBytes;

    // 3. RSA PKCS#1 서명 (DigestInfo 전체를 서명)
    final signer = PKCS1Encoding(RSAEngine());
    signer.init(true, PrivateKeyParameter<RSAPrivateKey>(privateKey));
    final signature = signer.process(digestInfoBytes);

    return base64.encode(signature);
  } catch (e) {
    print('📛 전자서명 생성 실패: $e');
    return null;
  }
}

  // PEM 형식의 개인 키를 RSAPrivateKey 객체로 파싱
  RSAPrivateKey _parsePrivateKeyFromPem(String privateKeyPem) {
    // PEM 헤더/푸터 제거 및 줄바꿈 제거
    String pemContent = privateKeyPem
        .replaceAll('-----BEGIN PRIVATE KEY-----', '')
        .replaceAll('-----END PRIVATE KEY-----', '')
        .replaceAll('-----BEGIN RSA PRIVATE KEY-----', '')
        .replaceAll('-----END RSA PRIVATE KEY-----', '')
        .replaceAll('\n', '')
        .replaceAll('\r', '')
        .trim();
    
    // Base64 디코드
    Uint8List keyBytes = base64.decode(pemContent);
    
    // PKCS#1 형식인지 PKCS#8 형식인지 확인
    bool isPkcs8 = privateKeyPem.contains('BEGIN PRIVATE KEY');
    
    if (isPkcs8) {
      // PKCS#8 형식 처리
      return _parsePkcs8PrivateKey(keyBytes);
    } else {
      // PKCS#1 형식 처리
      return _parsePkcs1PrivateKey(keyBytes);
    }
  }

  // PKCS#8 형식의 RSA 개인 키 파싱
  RSAPrivateKey _parsePkcs8PrivateKey(Uint8List keyBytes) {
    // ASN.1 파싱
    ASN1Parser parser = ASN1Parser(keyBytes);
    ASN1Sequence topLevelSeq = parser.nextObject() as ASN1Sequence;
    
    // PKCS#8 형식: PrivateKeyInfo
    // 0: version
    // 1: privateKeyAlgorithm
    // 2: privateKey (octet string)
    
    ASN1OctetString privateKeyOctet = topLevelSeq.elements[2] as ASN1OctetString;
    ASN1Parser privateKeyParser = ASN1Parser(privateKeyOctet.contentBytes());
    ASN1Sequence pkcs1PrivateKey = privateKeyParser.nextObject() as ASN1Sequence;
    
    return _parseRsaPrivateKeySequence(pkcs1PrivateKey);
  }

  // PKCS#1 형식의 RSA 개인 키 파싱
  RSAPrivateKey _parsePkcs1PrivateKey(Uint8List keyBytes) {
    ASN1Parser parser = ASN1Parser(keyBytes);
    ASN1Sequence privateKeySeq = parser.nextObject() as ASN1Sequence;
    
    return _parseRsaPrivateKeySequence(privateKeySeq);
  }

  // RSA 개인 키 ASN.1 시퀀스 파싱
  RSAPrivateKey _parseRsaPrivateKeySequence(ASN1Sequence sequence) {
    // RSA 개인 키 구조 (PKCS#1 RSAPrivateKey)
    // 0: version
    // 1: modulus (n)
    // 2: publicExponent (e)
    // 3: privateExponent (d)
    // 4: prime1 (p)
    // 5: prime2 (q)
    // 6: exponent1 (d mod (p-1))
    // 7: exponent2 (d mod (q-1))
    // 8: coefficient (q^-1 mod p)
    
    BigInt modulus = (sequence.elements[1] as ASN1Integer).valueAsBigInteger!;
    BigInt privateExponent = (sequence.elements[3] as ASN1Integer).valueAsBigInteger!;
    BigInt p = (sequence.elements[4] as ASN1Integer).valueAsBigInteger!;
    BigInt q = (sequence.elements[5] as ASN1Integer).valueAsBigInteger!;
    
    return RSAPrivateKey(modulus, privateExponent, p, q);
  }
}