import pytest

from fulcrypt import aead, aes, kdf, mac, modes

# =============================================================================
# Step 12: Encrypt-then-MAC  (slides 4c p13-14)
# =============================================================================
# Layout: nonce (16) || AES-128-CTR ciphertext || HMAC-SHA256 tag (32) over nonce || ciphertext.
# The nonce is random, so there are no fixed vectors: the tests take the output apart and
# rebuild each piece from the primitives of the earlier steps.

KEY = bytes(range(32))
NONCE_SIZE = 16
TAG_SIZE = 32


def split(data):
    return data[:NONCE_SIZE], data[NONCE_SIZE:-TAG_SIZE], data[-TAG_SIZE:]


def flip(data, i):
    out = bytearray(data)
    out[i] ^= 0x01
    return bytes(out)


class TestEtmDeriveKeys:
    def test_matches_hkdf(self):
        k_enc, k_mac = aead.etm_derive_keys(KEY)
        assert k_enc == kdf.hkdf(KEY, b"", b"encryption", 16)
        assert k_mac == kdf.hkdf(KEY, b"", b"authentication", 32)

    def test_sizes(self):
        k_enc, k_mac = aead.etm_derive_keys(KEY)
        assert type(k_enc) is bytes and type(k_mac) is bytes
        assert len(k_enc) == 16 and len(k_mac) == 32

    def test_keys_are_independent(self):
        # Never use the same key for encryption and authentication.
        k_enc, k_mac = aead.etm_derive_keys(KEY)
        assert k_enc != k_mac[:16]

    def test_different_master_keys(self):
        assert aead.etm_derive_keys(b"key one") != aead.etm_derive_keys(b"key two")


class TestEtmEncrypt:
    def test_layout_length(self):
        for length in (0, 1, 16, 33):
            assert len(aead.etm_encrypt(KEY, b"m" * length)) == NONCE_SIZE + length + TAG_SIZE

    def test_ciphertext_is_ctr_under_nonce(self):
        msg = b"attack at dawn, bring snacks"
        nonce, ct, _ = split(aead.etm_encrypt(KEY, msg))
        k_enc, _ = aead.etm_derive_keys(KEY)
        assert ct == modes.ctr_encrypt(msg, k_enc, nonce)

    def test_tag_is_hmac_of_nonce_and_ciphertext(self):
        # Encrypt-then-MAC: the tag covers what's sent (nonce || ciphertext), not the plaintext.
        nonce, ct, tag = split(aead.etm_encrypt(KEY, b"attack at dawn"))
        _, k_mac = aead.etm_derive_keys(KEY)
        assert tag == mac.hmac_sha256(k_mac, nonce + ct)

    def test_random_nonce(self):
        # Same key, same message: a fresh nonce makes every ciphertext different.
        outputs = {aead.etm_encrypt(KEY, b"same message") for _ in range(5)}
        assert len(outputs) == 5
        assert len({split(o)[0] for o in outputs}) == 5

    def test_ciphertext_hides_plaintext(self):
        msg = b"secret!!" * 4
        _, ct, _ = split(aead.etm_encrypt(KEY, msg))
        assert ct != msg

    def test_returns_bytes(self):
        assert type(aead.etm_encrypt(KEY, b"abc")) is bytes

    def test_accepts_bytearray(self):
        data = aead.etm_encrypt(bytearray(KEY), bytearray(b"abc"))
        assert aead.etm_decrypt(KEY, data) == b"abc"


class TestEtmDecrypt:
    @pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 100, 1000])
    def test_round_trip(self, length):
        msg = bytes((7 * i + 1) % 256 for i in range(length))
        assert aead.etm_decrypt(KEY, aead.etm_encrypt(KEY, msg)) == msg

    @pytest.mark.parametrize("key", [b"short", bytes(16), bytes(range(64))])
    def test_any_master_key_length(self, key):
        # The master key only feeds HKDF, so any length works.
        assert aead.etm_decrypt(key, aead.etm_encrypt(key, b"hello")) == b"hello"

    def test_returns_bytes(self):
        assert type(aead.etm_decrypt(KEY, aead.etm_encrypt(KEY, b"abc"))) is bytes

    def test_accepts_bytearray(self):
        data = aead.etm_encrypt(KEY, b"abc")
        assert aead.etm_decrypt(bytearray(KEY), bytearray(data)) == b"abc"


class TestEtmRejectsTampering:
    MSG = b"amount=100&to=bob"

    def data(self):
        return aead.etm_encrypt(KEY, self.MSG)

    @pytest.mark.parametrize("where", ["nonce", "ciphertext", "tag"])
    @pytest.mark.parametrize("position", ["first", "last"])
    def test_any_flipped_bit(self, where, position):
        data = self.data()
        start = {"nonce": 0, "ciphertext": NONCE_SIZE, "tag": len(data) - TAG_SIZE}[where]
        end = {"nonce": NONCE_SIZE, "ciphertext": len(data) - TAG_SIZE, "tag": len(data)}[where]
        i = start if position == "first" else end - 1
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, flip(data, i))

    def test_ctr_malleability_is_caught(self):
        # CTR alone lets an attacker turn "100" into "900" by XORing the ciphertext.
        # Here the change is detected because the tag no longer matches.
        data = bytearray(self.data())
        i = NONCE_SIZE + self.MSG.index(b"1")
        data[i] ^= ord("1") ^ ord("9")
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, bytes(data))

    def test_wrong_key(self):
        with pytest.raises(ValueError):
            aead.etm_decrypt(b"wrong key", self.data())

    def test_tag_from_another_message(self):
        nonce, ct, _ = split(self.data())
        _, _, other_tag = split(aead.etm_encrypt(KEY, b"amount=999&to=eve"))
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, nonce + ct + other_tag)

    def test_swapped_nonce(self):
        # Re-pairing a ciphertext with another nonce changes the decryption: the tag forbids it.
        _, ct, tag = split(self.data())
        other_nonce, _, _ = split(aead.etm_encrypt(KEY, self.MSG))
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, other_nonce + ct + tag)

    @pytest.mark.parametrize("cut", [1, 16, 32])
    def test_truncated(self, cut):
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, self.data()[:-cut])

    def test_extended(self):
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, self.data() + b"\x00")

    def test_dropped_ciphertext_byte(self):
        data = self.data()
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, data[:NONCE_SIZE] + data[NONCE_SIZE + 1 :])

    @pytest.mark.parametrize("length", [0, 1, 47])
    def test_too_short(self, length):
        # Anything shorter than nonce + tag (48 bytes) can't be valid.
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, bytes(length))

    def test_empty_message_tag_still_checked(self):
        data = aead.etm_encrypt(KEY, b"")
        assert len(data) == 48
        with pytest.raises(ValueError):
            aead.etm_decrypt(KEY, flip(data, 47))


# =============================================================================
# Step 13: GF(2^128) for GCM, and GCM's reflected bit order  (SP 800-38D section 6.3)
# =============================================================================
# GCM reads a 16-byte block as a polynomial with the coefficient of x^0 in the MOST significant
# bit of byte 0, and x^127 in the least significant bit of byte 15: the reverse of GF2n's bits.


def blk(hex_string):
    return bytes.fromhex(hex_string)


ONE = blk("80000000000000000000000000000000")  # x^0
X1 = blk("40000000000000000000000000000000")  # x^1
X127 = blk("00000000000000000000000000000001")  # x^127


class TestGcmField:
    def test_modulus(self):
        assert aead.GCM_FIELD.modulus.bits == (1 << 128) | 0x87

    def test_one(self):
        assert aead.block_to_gcm(ONE) == aead.GCM_FIELD(1)

    def test_x(self):
        assert aead.block_to_gcm(X1) == aead.GCM_FIELD(0b10)

    def test_x127(self):
        assert aead.block_to_gcm(X127) == aead.GCM_FIELD(1 << 127)

    def test_zero(self):
        assert aead.block_to_gcm(bytes(16)) == aead.GCM_FIELD(0)

    def test_byte_is_reversed_too(self):
        # 0x01 in byte 0 is bit 7 of the block: x^7, not x^0.
        assert aead.block_to_gcm(blk("01" + "00" * 15)) == aead.GCM_FIELD(1 << 7)

    def test_e1_is_the_reduction(self):
        # x * x^127 = x^128 = x^7 + x^2 + x + 1, which in GCM's order is the block e1 00..00.
        product = aead.block_to_gcm(X1) * aead.block_to_gcm(X127)
        assert product == aead.GCM_FIELD(0x87)
        assert aead.gcm_to_block(product) == blk("e1000000000000000000000000000000")

    def test_gcm_test_case_2_ghash(self):
        # GCM spec test case 2 (key 0^128, one zero plaintext block):
        # GHASH = ((C * H) + L) * H with L = len(A) || len(C) in bits = 0 || 128.
        h = aead.block_to_gcm(blk("66e94bd4ef8a2c3b884cfa59ca342b2e"))
        c = aead.block_to_gcm(blk("0388dace60b6a392f328c2b971b2fe78"))
        lengths = aead.block_to_gcm((0).to_bytes(8, "big") + (128).to_bytes(8, "big"))
        assert aead.gcm_to_block((c * h + lengths) * h).hex() == "f38cbb1ad69223dcc3457ae5b6b0f885"

    def test_addition_is_block_xor(self):
        a, b = blk("0123456789abcdef0123456789abcdef"), blk("fedcba9876543210fedcba9876543210")
        total = aead.block_to_gcm(a) + aead.block_to_gcm(b)
        assert aead.gcm_to_block(total) == bytes(x ^ y for x, y in zip(a, b, strict=True))

    @pytest.mark.parametrize(
        "block",
        [ONE, X1, X127, bytes(16), b"\xff" * 16, blk("66e94bd4ef8a2c3b884cfa59ca342b2e")],
    )
    def test_round_trip(self, block):
        assert aead.gcm_to_block(aead.block_to_gcm(block)) == block

    def test_gcm_to_block_returns_16_bytes(self):
        out = aead.gcm_to_block(aead.GCM_FIELD(1))
        assert type(out) is bytes
        assert out == ONE

    def test_accepts_bytearray(self):
        assert aead.block_to_gcm(bytearray(X1)) == aead.block_to_gcm(X1)

    @pytest.mark.parametrize("length", [0, 15, 17, 32])
    def test_rejects_wrong_block_size(self, length):
        with pytest.raises(ValueError):
            aead.block_to_gcm(bytes(length))


# =============================================================================
# Step 14: GHASH  (SP 800-38D section 6.4; vectors from the GCM spec, McGrew & Viega)
# =============================================================================
# Y_0 = 0, Y_i = (Y_{i-1} + X_i) * H over: AAD zero-padded to 16 bytes, ciphertext zero-padded,
# then one block len(AAD) || len(C), both in BITS as 64-bit big-endian. Output: the last Y.

H_ZERO_KEY = blk("66e94bd4ef8a2c3b884cfa59ca342b2e")  # AES(0^128, 0^128)
H_TC3 = blk("b83b533708bf535d0aa6e52980d53b78")  # AES(feffe992..., 0^128)
C_TC3 = blk(
    "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e"
    "21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091473f5985"
)
A_TC4 = blk("feedfacedeadbeeffeedfacedeadbeefabaddad2")


def pattern(n, step):
    return bytes((step * i + 1) % 256 for i in range(n))


def lengths_block(aad, ciphertext):
    return (8 * len(aad)).to_bytes(8, "big") + (8 * len(ciphertext)).to_bytes(8, "big")


class TestGhash:
    @pytest.mark.parametrize(
        "h, aad, ciphertext, expected",
        [
            pytest.param(H_ZERO_KEY, b"", b"", "00000000000000000000000000000000", id="TC1"),
            pytest.param(
                H_ZERO_KEY,
                b"",
                blk("0388dace60b6a392f328c2b971b2fe78"),
                "f38cbb1ad69223dcc3457ae5b6b0f885",
                id="TC2",
            ),
            pytest.param(H_TC3, b"", C_TC3, "7f1b32b81b820d02614f8895ac1d4eac", id="TC3"),
            pytest.param(  # 20-byte AAD and 60-byte ciphertext: both need padding
                H_TC3, A_TC4, C_TC3[:60], "698e57f70e6ecc7fd9463b7260a9ae5f", id="TC4"
            ),
        ],
    )
    def test_gcm_spec(self, h, aad, ciphertext, expected):
        assert aead.ghash(h, aad, ciphertext).hex() == expected

    # Extra padding cases (computed with an independent implementation of SP 800-38D Alg. 1).
    @pytest.mark.parametrize(
        "aad_len, c_len, expected",
        [
            (20, 0, "185df651e0c6cbc79c719d3183eaa624"),  # AAD only (GMAC)
            (1, 0, "e75eec3c61fee545337e33d02023d0d7"),
            (0, 1, "d04d539954afcba6c45d33779362c3c3"),
            (16, 16, "1958da75216a6b8d2ea5ca3ae333c5dc"),  # no padding at all
            (17, 33, "0a57586cffe16f7306ebc376c015e635"),  # one byte over, twice
            (48, 15, "11596e0b3ee7172200c2135ccf7a53ba"),
        ],
    )
    def test_padding_cases(self, aad_len, c_len, expected):
        assert aead.ghash(H_TC3, pattern(aad_len, 5), pattern(c_len, 7)).hex() == expected

    def test_h_is_aes_of_zero_block(self):
        # Where H comes from in GCM (computed by the caller; ghash takes it as input).
        assert aes.encrypt_block(bytes(16), bytes(16)) == H_ZERO_KEY

    def test_one_block_by_hand(self):
        # One ciphertext block, no AAD: ((C * H) + L) * H.
        c = pattern(16, 3)
        h, cg = aead.block_to_gcm(H_TC3), aead.block_to_gcm(c)
        lg = aead.block_to_gcm(lengths_block(b"", c))
        assert aead.ghash(H_TC3, b"", c) == aead.gcm_to_block((cg * h + lg) * h)

    def test_aad_block_comes_first(self):
        # One AAD block and one ciphertext block: (((A * H) + C) * H + L) * H.
        a, c = pattern(16, 5), pattern(16, 7)
        h = aead.block_to_gcm(H_TC3)
        y = aead.block_to_gcm(a) * h
        y = (y + aead.block_to_gcm(c)) * h
        y = (y + aead.block_to_gcm(lengths_block(a, c))) * h
        assert aead.ghash(H_TC3, a, c) == aead.gcm_to_block(y)

    def test_aad_and_ciphertext_not_interchangeable(self):
        # Same bytes, different role: the lengths block tells them apart.
        x = pattern(32, 9)
        assert aead.ghash(H_TC3, x, b"") != aead.ghash(H_TC3, b"", x)

    def test_padding_is_not_ambiguous(self):
        # Zero padding alone would make these equal; the lengths block separates them.
        c = pattern(15, 9)
        assert aead.ghash(H_TC3, b"", c) != aead.ghash(H_TC3, b"", c + b"\x00")

    def test_lengths_are_in_bits(self):
        # With all-zero data only the lengths block matters: Y = L * H.
        expected = aead.block_to_gcm(lengths_block(b"", bytes(5))) * aead.block_to_gcm(H_TC3)
        assert aead.ghash(H_TC3, b"", bytes(5)) == aead.gcm_to_block(expected)

    def test_zero_h_gives_zero(self):
        assert aead.ghash(bytes(16), pattern(20, 5), pattern(40, 7)) == bytes(16)

    def test_returns_16_bytes(self):
        out = aead.ghash(H_TC3, b"aad", b"ciphertext")
        assert type(out) is bytes
        assert len(out) == 16

    def test_accepts_bytearray(self):
        args = (H_TC3, A_TC4, C_TC3[:60])
        assert aead.ghash(*(bytearray(a) for a in args)) == aead.ghash(*args)

    @pytest.mark.parametrize("length", [0, 15, 17, 32])
    def test_rejects_wrong_h_size(self, length):
        with pytest.raises(ValueError):
            aead.ghash(bytes(length), b"", b"")


# =============================================================================
# Step 15: AES-GCM  (SP 800-38D section 7; vectors from the GCM spec, McGrew & Viega)
# =============================================================================
# 96-bit IV only: J0 = IV || 0^31 || 1. CTR from inc32(J0); tag T = AES(K, J0) xor GHASH(H, A, C).

GCM_K = "feffe9928665731c6d6a8f9467308308"
GCM_P = blk(
    "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"
    "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255"
)
GCM_IV = blk("cafebabefacedbaddecaf888")
GCM_A = A_TC4


def gcm_case(name, key_hex, plaintext, iv, aad, c_hex, t_hex):
    return pytest.param(blk(key_hex), iv, plaintext, aad, blk(c_hex), blk(t_hex), id=name)


GCM_SPEC = [
    # AES-128
    gcm_case("TC1", "00" * 16, b"", bytes(12), b"", "", "58e2fccefa7e3061367f1d57a4e7455a"),
    gcm_case(
        "TC2",
        "00" * 16,
        bytes(16),
        bytes(12),
        b"",
        "0388dace60b6a392f328c2b971b2fe78",
        "ab6e47d42cec13bdf53a67b21257bddf",
    ),
    gcm_case("TC3", GCM_K, GCM_P, GCM_IV, b"", C_TC3.hex(), "4d5c2af327cd64a62cf35abd2ba6fab4"),
    gcm_case(
        "TC4",
        GCM_K,
        GCM_P[:60],
        GCM_IV,
        GCM_A,
        C_TC3[:60].hex(),
        "5bc94fbc3221a5db94fae95ae7121a47",
    ),
    # AES-192
    gcm_case("TC7", "00" * 24, b"", bytes(12), b"", "", "cd33b28ac773f74ba00ed1f312572435"),
    gcm_case(
        "TC8",
        "00" * 24,
        bytes(16),
        bytes(12),
        b"",
        "98e7247c07f0fe411c267e4384b0f600",
        "2ff58d80033927ab8ef4d4587514f0fb",
    ),
    gcm_case(
        "TC9",
        GCM_K + GCM_K[:16],
        GCM_P,
        GCM_IV,
        b"",
        "3980ca0b3c00e841eb06fac4872a2757859e1ceaa6efd984628593b40ca1e19c"
        "7d773d00c144c525ac619d18c84a3f4718e2448b2fe324d9ccda2710acade256",
        "9924a7c8587336bfb118024db8674a14",
    ),
    gcm_case(
        "TC10",
        GCM_K + GCM_K[:16],
        GCM_P[:60],
        GCM_IV,
        GCM_A,
        "3980ca0b3c00e841eb06fac4872a2757859e1ceaa6efd984628593b40ca1e19c"
        "7d773d00c144c525ac619d18c84a3f4718e2448b2fe324d9ccda2710",
        "2519498e80f1478f37ba55bd6d27618c",
    ),
    # AES-256
    gcm_case("TC13", "00" * 32, b"", bytes(12), b"", "", "530f8afbc74536b9a963b4f1c4cb738b"),
    gcm_case(
        "TC14",
        "00" * 32,
        bytes(16),
        bytes(12),
        b"",
        "cea7403d4d606b6e074ec5d3baf39d18",
        "d0d1c8a799996bf0265b98b5d48ab919",
    ),
    gcm_case(
        "TC15",
        GCM_K * 2,
        GCM_P,
        GCM_IV,
        b"",
        "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"
        "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662898015ad",
        "b094dac5d93471bdec1a502270e3cc6c",
    ),
    gcm_case(
        "TC16",
        GCM_K * 2,
        GCM_P[:60],
        GCM_IV,
        GCM_A,
        "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"
        "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662",
        "76fc6ece0f4e1768cddf8853bb2d551b",
    ),
]

GK = blk(GCM_K)


class TestGcmEncrypt:
    @pytest.mark.parametrize("key, iv, plaintext, aad, ciphertext, tag", GCM_SPEC)
    def test_gcm_spec(self, key, iv, plaintext, aad, ciphertext, tag):
        assert aead.gcm_encrypt(key, iv, plaintext, aad) == (ciphertext, tag)

    def test_ciphertext_is_ctr_from_inc32_j0(self):
        # Keystream block 1 is AES(K, IV || 00000002): J0 = IV || 00000001 is kept for the tag.
        j0 = GCM_IV + (1).to_bytes(4, "big")
        start = GCM_IV + (2).to_bytes(4, "big")
        ciphertext, _ = aead.gcm_encrypt(GK, GCM_IV, GCM_P)
        assert ciphertext == modes.ctr_encrypt(GCM_P, GK, start, counter_bits=32)
        assert ciphertext[:16] != modes.ctr_encrypt(GCM_P[:16], GK, j0, counter_bits=32)

    def test_tag_is_ghash_masked_with_aes_j0(self):
        ciphertext, tag = aead.gcm_encrypt(GK, GCM_IV, GCM_P[:60], GCM_A)
        h = aes.encrypt_block(bytes(16), GK)
        s = aes.encrypt_block(GCM_IV + (1).to_bytes(4, "big"), GK)
        expected = bytes(x ^ y for x, y in zip(s, aead.ghash(h, GCM_A, ciphertext), strict=True))
        assert tag == expected

    def test_aad_is_authenticated_not_encrypted(self):
        c1, t1 = aead.gcm_encrypt(GK, GCM_IV, b"hello", b"header one")
        c2, t2 = aead.gcm_encrypt(GK, GCM_IV, b"hello", b"header two")
        assert c1 == c2
        assert t1 != t2

    def test_default_aad_is_empty(self):
        assert aead.gcm_encrypt(GK, GCM_IV, b"hi") == aead.gcm_encrypt(GK, GCM_IV, b"hi", b"")

    def test_aad_only_is_gmac(self):
        # No plaintext: an authentication tag over the AAD alone.
        ciphertext, tag = aead.gcm_encrypt(GK, GCM_IV, b"", GCM_A)
        assert ciphertext == b""
        assert len(tag) == 16

    def test_nonce_reuse_leaks_xor_of_plaintexts(self):
        # Why an IV must never repeat under one key: it's CTR underneath.
        p1, p2 = b"attack at dawn!!", b"retreat at dusk!"
        c1, _ = aead.gcm_encrypt(GK, GCM_IV, p1)
        c2, _ = aead.gcm_encrypt(GK, GCM_IV, p2)
        assert bytes(a ^ b for a, b in zip(c1, c2, strict=True)) == bytes(
            a ^ b for a, b in zip(p1, p2, strict=True)
        )

    def test_returns_bytes_pair(self):
        ciphertext, tag = aead.gcm_encrypt(GK, GCM_IV, b"abc")
        assert type(ciphertext) is bytes and type(tag) is bytes
        assert len(ciphertext) == 3 and len(tag) == 16

    def test_accepts_bytearray(self):
        args = (GK, GCM_IV, GCM_P[:60], GCM_A)
        assert aead.gcm_encrypt(*(bytearray(a) for a in args)) == aead.gcm_encrypt(*args)

    @pytest.mark.parametrize("iv_length", [0, 8, 11, 13, 16])
    def test_rejects_non_96_bit_iv(self, iv_length):
        with pytest.raises(ValueError):
            aead.gcm_encrypt(GK, bytes(iv_length), b"abc")


class TestGcmDecrypt:
    @pytest.mark.parametrize("key, iv, plaintext, aad, ciphertext, tag", GCM_SPEC)
    def test_gcm_spec(self, key, iv, plaintext, aad, ciphertext, tag):
        assert aead.gcm_decrypt(key, iv, ciphertext, tag, aad) == plaintext

    @pytest.mark.parametrize(
        "p_len, a_len", [(0, 0), (1, 0), (15, 3), (16, 16), (17, 20), (100, 0)]
    )
    def test_round_trip(self, p_len, a_len):
        p, a = pattern(p_len, 7), pattern(a_len, 5)
        ciphertext, tag = aead.gcm_encrypt(GK, GCM_IV, p, a)
        assert aead.gcm_decrypt(GK, GCM_IV, ciphertext, tag, a) == p

    def test_returns_bytes(self):
        ciphertext, tag = aead.gcm_encrypt(GK, GCM_IV, b"abc")
        assert type(aead.gcm_decrypt(GK, GCM_IV, ciphertext, tag)) is bytes

    def test_accepts_bytearray(self):
        ciphertext, tag = aead.gcm_encrypt(GK, GCM_IV, b"abc", b"hdr")
        args = (GK, GCM_IV, ciphertext, tag, b"hdr")
        assert aead.gcm_decrypt(*(bytearray(a) for a in args)) == b"abc"

    @pytest.mark.parametrize("iv_length", [0, 8, 16])
    def test_rejects_non_96_bit_iv(self, iv_length):
        ciphertext, tag = aead.gcm_encrypt(GK, GCM_IV, b"abc")
        with pytest.raises(ValueError):
            aead.gcm_decrypt(GK, bytes(iv_length), ciphertext, tag)


class TestGcmRejectsTampering:
    P = b"amount=100&to=bob"
    A = b"msg-id: 42"

    def sealed(self):
        ciphertext, tag = aead.gcm_encrypt(GK, GCM_IV, self.P, self.A)
        return ciphertext, tag

    @pytest.mark.parametrize("i", [0, 11])
    def test_flipped_iv(self, i):
        ciphertext, tag = self.sealed()
        with pytest.raises(ValueError):
            aead.gcm_decrypt(GK, flip(GCM_IV, i), ciphertext, tag, self.A)

    @pytest.mark.parametrize("i", [0, 9])
    def test_flipped_aad(self, i):
        ciphertext, tag = self.sealed()
        with pytest.raises(ValueError):
            aead.gcm_decrypt(GK, GCM_IV, ciphertext, tag, flip(self.A, i))

    def test_dropped_aad(self):
        ciphertext, tag = self.sealed()
        with pytest.raises(ValueError):
            aead.gcm_decrypt(GK, GCM_IV, ciphertext, tag)

    @pytest.mark.parametrize("i", [0, 7, 16])
    def test_flipped_ciphertext(self, i):
        ciphertext, tag = self.sealed()
        with pytest.raises(ValueError):
            aead.gcm_decrypt(GK, GCM_IV, flip(ciphertext, i), tag, self.A)

    def test_ctr_malleability_is_caught(self):
        # "100" -> "900" by XORing the ciphertext, as in the Encrypt-then-MAC tests.
        ciphertext, tag = self.sealed()
        i = self.P.index(b"1")
        forged = bytearray(ciphertext)
        forged[i] ^= ord("1") ^ ord("9")
        with pytest.raises(ValueError):
            aead.gcm_decrypt(GK, GCM_IV, bytes(forged), tag, self.A)

    @pytest.mark.parametrize("i", [0, 15])
    def test_flipped_tag(self, i):
        ciphertext, tag = self.sealed()
        with pytest.raises(ValueError):
            aead.gcm_decrypt(GK, GCM_IV, ciphertext, flip(tag, i), self.A)

    @pytest.mark.parametrize("length", [0, 12, 15])
    def test_truncated_tag(self, length):
        ciphertext, tag = self.sealed()
        with pytest.raises(ValueError):
            aead.gcm_decrypt(GK, GCM_IV, ciphertext, tag[:length], self.A)

    def test_truncated_ciphertext(self):
        ciphertext, tag = self.sealed()
        with pytest.raises(ValueError):
            aead.gcm_decrypt(GK, GCM_IV, ciphertext[:-1], tag, self.A)

    def test_wrong_key(self):
        ciphertext, tag = self.sealed()
        with pytest.raises(ValueError):
            aead.gcm_decrypt(bytes(16), GCM_IV, ciphertext, tag, self.A)
