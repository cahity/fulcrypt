import hashlib
import hmac

import pytest

from fulcrypt import aes, hashes, mac, modes

# =============================================================================
# Step 7: HMAC-SHA256  (FIPS 198-1, RFC 2104, test vectors RFC 4231)
# =============================================================================

RFC4231_MSG_7 = (
    b"This is a test using a larger than block-size key and a larger than block-size data. "
    b"The key needs to be hashed before being used by the HMAC algorithm."
)


class TestHmacSha256:
    # RFC 4231 section 4, test cases 1-4, 6, 7 (case 5 is truncation, below).
    @pytest.mark.parametrize(
        "key, message, tag",
        [
            (
                b"\x0b" * 20,
                b"Hi There",
                "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
            ),
            (
                b"Jefe",
                b"what do ya want for nothing?",
                "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843",
            ),
            (
                b"\xaa" * 20,
                b"\xdd" * 50,
                "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe",
            ),
            (
                bytes(range(1, 26)),
                b"\xcd" * 50,
                "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b",
            ),
            (  # 131-byte key: longer than the block, hashed first
                b"\xaa" * 131,
                b"Test Using Larger Than Block-Size Key - Hash Key First",
                "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54",
            ),
            (  # long key and a multi-block message
                b"\xaa" * 131,
                RFC4231_MSG_7,
                "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2",
            ),
        ],
    )
    def test_rfc4231(self, key, message, tag):
        assert mac.hmac_sha256(key, message).hex() == tag

    def test_rfc4231_case5_truncated(self):
        # The RFC gives only the first 128 bits of this tag.
        tag = mac.hmac_sha256(b"\x0c" * 20, b"Test With Truncation")
        assert tag[:16].hex() == "a3b6167473100ee06e0c796c2955552b"

    # Key lengths around the block size: padded (< 64), used as is (64), hashed first (> 64).
    @pytest.mark.parametrize("key_length", [0, 1, 31, 32, 33, 63, 64, 65, 100, 131])
    def test_matches_stdlib_every_key_case(self, key_length):
        key = bytes((11 * i + 5) % 256 for i in range(key_length))
        msg = b"The quick brown fox jumps over the lazy dog"
        assert mac.hmac_sha256(key, msg) == hmac.new(key, msg, hashlib.sha256).digest()

    @pytest.mark.parametrize("msg_length", [0, 1, 55, 56, 64, 119, 120, 200])
    def test_matches_stdlib_every_message_length(self, msg_length):
        key = b"key"
        msg = bytes((3 * i + 1) % 256 for i in range(msg_length))
        assert mac.hmac_sha256(key, msg) == hmac.new(key, msg, hashlib.sha256).digest()

    def test_returns_32_bytes(self):
        tag = mac.hmac_sha256(b"key", b"message")
        assert type(tag) is bytes
        assert len(tag) == 32

    def test_long_key_equals_its_hash(self):
        # A key longer than 64 bytes is replaced by its SHA-256 digest.
        key = b"k" * 100
        assert mac.hmac_sha256(key, b"m") == mac.hmac_sha256(hashes.sha256(key), b"m")

    def test_key_padding_is_zeros(self):
        # Short keys are padded with zero bytes, so trailing zeros don't change the tag.
        assert mac.hmac_sha256(b"key", b"m") == mac.hmac_sha256(b"key\x00\x00", b"m")

    def test_is_not_plain_hash_of_key_and_message(self):
        assert mac.hmac_sha256(b"key", b"m") != hashes.sha256(b"key" + b"m")

    def test_length_extension_fails(self):
        # Step 6 forges sha256(key || m || ...), but not HMAC: the tag is the outer hash.
        key, msg = b"secret key", b"user=alice"
        tag = mac.hmac_sha256(key, msg)
        ext, forged = hashes.sha256_extend(tag, 64 + len(msg), b"&role=admin")
        assert forged != mac.hmac_sha256(key, msg + ext)

    def test_different_keys_different_tags(self):
        assert mac.hmac_sha256(b"key1", b"m") != mac.hmac_sha256(b"key2", b"m")

    def test_accepts_bytearray(self):
        assert mac.hmac_sha256(bytearray(b"key"), bytearray(b"m")) == mac.hmac_sha256(b"key", b"m")


class TestConstantTimeEqual:
    def test_equal(self):
        assert mac.constant_time_equal(b"abc", b"abc") is True

    def test_empty(self):
        assert mac.constant_time_equal(b"", b"") is True

    @pytest.mark.parametrize("other", [b"xbc", b"axc", b"abx"])  # first, middle, last byte
    def test_one_byte_differs(self, other):
        assert mac.constant_time_equal(b"abc", other) is False

    def test_one_bit_differs(self):
        assert mac.constant_time_equal(b"\x00" * 32, b"\x00" * 31 + b"\x01") is False

    @pytest.mark.parametrize("other", [b"", b"ab", b"abcd"])
    def test_different_lengths(self, other):
        assert mac.constant_time_equal(b"abc", other) is False

    def test_matches_stdlib(self):
        pairs = [(b"a" * 32, b"a" * 32), (b"a" * 32, b"a" * 31 + b"b"), (b"\xff", b"\xfe")]
        for a, b in pairs:
            assert mac.constant_time_equal(a, b) == hmac.compare_digest(a, b)


class TestHmacSha256Verify:
    KEY = b"secret key"
    MSG = b"amount=100&to=bob"

    def tag(self):
        return mac.hmac_sha256(self.KEY, self.MSG)

    def test_accepts_valid_tag(self):
        assert mac.hmac_sha256_verify(self.KEY, self.MSG, self.tag()) is True

    def test_rejects_wrong_key(self):
        assert mac.hmac_sha256_verify(b"other key", self.MSG, self.tag()) is False

    def test_rejects_modified_message(self):
        assert mac.hmac_sha256_verify(self.KEY, b"amount=900&to=bob", self.tag()) is False

    def test_rejects_flipped_tag_bit(self):
        tag = bytearray(self.tag())
        tag[-1] ^= 0x01
        assert mac.hmac_sha256_verify(self.KEY, self.MSG, bytes(tag)) is False

    @pytest.mark.parametrize("length", [0, 16, 31])
    def test_rejects_truncated_tag(self, length):
        assert mac.hmac_sha256_verify(self.KEY, self.MSG, self.tag()[:length]) is False

    def test_rejects_extended_tag(self):
        assert mac.hmac_sha256_verify(self.KEY, self.MSG, self.tag() + b"\x00") is False


# =============================================================================
# Step 8: CBC-MAC and its forgery  (slides 4b p16-17)
# =============================================================================

KEY_128 = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")  # SP 800-38A key
ZERO_IV = bytes(16)


def xor(a, b):
    return bytes(x ^ y for x, y in zip(a, b, strict=True))


class TestCbcMac:
    def test_one_block_is_aes(self):
        # FIPS-197 Appendix C.1: with one block, CBC-MAC is just AES of that block.
        key = bytes(range(16))
        block = bytes.fromhex("00112233445566778899aabbccddeeff")
        assert mac.cbc_mac(key, block).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"

    def test_two_blocks_by_hand(self):
        m1, m2 = b"A" * 16, b"B" * 16
        expected = aes.encrypt_block(xor(aes.encrypt_block(m1, KEY_128), m2), KEY_128)
        assert mac.cbc_mac(KEY_128, m1 + m2) == expected

    @pytest.mark.parametrize("blocks", [1, 2, 3, 4, 10])
    def test_is_last_block_of_cbc_with_zero_iv(self, blocks):
        msg = bytes((9 * i + 4) % 256 for i in range(16 * blocks))
        assert mac.cbc_mac(KEY_128, msg) == modes.cbc_encrypt(msg, KEY_128, ZERO_IV)[-16:]

    @pytest.mark.parametrize("key_length", [16, 24, 32])
    def test_all_key_sizes(self, key_length):
        key = bytes(range(key_length))
        msg = b"sixteen byte msg" * 3
        assert mac.cbc_mac(key, msg) == modes.cbc_encrypt(msg, key, ZERO_IV)[-16:]

    def test_returns_16_bytes(self):
        tag = mac.cbc_mac(KEY_128, b"x" * 32)
        assert type(tag) is bytes
        assert len(tag) == 16

    def test_every_block_matters(self):
        msg = bytearray(b"x" * 48)
        base = mac.cbc_mac(KEY_128, bytes(msg))
        for i in (0, 20, 47):  # first, middle, last block
            changed = bytearray(msg)
            changed[i] ^= 0x01
            assert mac.cbc_mac(KEY_128, bytes(changed)) != base

    def test_accepts_bytearray(self):
        assert mac.cbc_mac(bytearray(KEY_128), bytearray(b"y" * 16)) == mac.cbc_mac(
            KEY_128, b"y" * 16
        )

    @pytest.mark.parametrize("length", [1, 15, 17, 31, 33])
    def test_rejects_unaligned(self, length):
        with pytest.raises(ValueError):
            mac.cbc_mac(KEY_128, b"z" * length)

    def test_rejects_empty(self):
        # No blocks means no last block: there is no tag to return.
        with pytest.raises(ValueError):
            mac.cbc_mac(KEY_128, b"")


class TestCbcMacForgery:
    """The attacker gets tags from a MAC oracle and forges a tag for a new message."""

    def oracle(self, message):
        return mac.cbc_mac(KEY_128, message)  # the attacker can query, but not see KEY_128

    def test_slides_forgery(self):
        # Slides: ask for t' = MAC(m'), then t = MAC(t' xor m''). Then t is also the tag
        # of m' || m'', a message never queried: the second block's input is t' xor m'' again.
        m1, m2 = b"pay alice $10.00", b"pay mallory $999"
        t1 = self.oracle(m1)
        t = self.oracle(xor(t1, m2))
        forged = m1 + m2
        assert mac.cbc_mac(KEY_128, forged) == t

    def test_splice_two_messages(self):
        # Given (m1, t1) and (m2, t2), m1 || (m2[0] xor t1) || m2[1:] has tag t2.
        m1 = b"first message, two blocks long!!"
        m2 = b"second message is exactly three blocks long, 48 bytes in total."[:48]
        t1, t2 = self.oracle(m1), self.oracle(m2)
        forged = m1 + xor(m2[:16], t1) + m2[16:]
        assert forged not in (m1, m2)
        assert mac.cbc_mac(KEY_128, forged) == t2

    def test_extend_with_zero_xor(self):
        # m || (t xor m) has the same tag t: the second block's input repeats the first's.
        m = b"transfer 100 EUR"
        t = self.oracle(m)
        assert mac.cbc_mac(KEY_128, m + xor(t, m)) == t


# =============================================================================
# Step 9: CMAC  (NIST SP 800-38B, test vectors RFC 4493 / SP 800-38B Appendix D)
# =============================================================================

# The SP 800-38A plaintext; the examples MAC its first 0, 16, 40 and 64 bytes.
CMAC_MSG = bytes.fromhex(
    "6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51"
    "30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710"
)
KEY_192 = bytes.fromhex("8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b")
KEY_256 = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")


class TestCmacSubkeys:
    def test_rfc4493(self):
        # L = AES(K, 0) = 7df76b0c...; its top bit is 0, so K1 = L << 1 only.
        # K1's top bit is 1, so K2 = (K1 << 1) xor 0x87: both branches of the doubling.
        k1, k2 = mac.cmac_subkeys(KEY_128)
        assert k1.hex() == "fbeed618357133667c85e08f7236a8de"
        assert k2.hex() == "f7ddac306ae266ccf90bc11ee46d513b"

    def test_returns_two_16_byte_strings(self):
        k1, k2 = mac.cmac_subkeys(KEY_128)
        assert type(k1) is bytes and type(k2) is bytes
        assert len(k1) == 16 and len(k2) == 16

    @pytest.mark.parametrize("key", [KEY_192, KEY_256])
    def test_other_key_sizes(self, key):
        k1, k2 = mac.cmac_subkeys(key)
        assert len(k1) == 16 and len(k2) == 16
        assert k1 != k2

    def test_k1_is_l_doubled(self):
        # Check the doubling rule directly on L = AES(K, 0^128), for a key where L's top bit is 1.
        key = b"\x01" * 16
        l_int = int.from_bytes(aes.encrypt_block(bytes(16), key), "big")  # b6aeaffa...
        assert l_int >> 127 == 1  # this key exercises the reduction branch in K1
        expected = ((l_int << 1) ^ 0x87) & ((1 << 128) - 1)
        assert mac.cmac_subkeys(key)[0] == expected.to_bytes(16, "big")


class TestCmac:
    # RFC 4493 section 4 (AES-128) and SP 800-38B Appendix D.2 / D.3 (AES-192 / AES-256).
    @pytest.mark.parametrize(
        "key, length, tag",
        [
            (KEY_128, 0, "bb1d6929e95937287fa37d129b756746"),
            (KEY_128, 16, "070a16b46b4d4144f79bdd9dd04a287c"),
            (KEY_128, 40, "dfa66747de9ae63030ca32611497c827"),
            (KEY_128, 64, "51f0bebf7e3b9d92fc49741779363cfe"),
            (KEY_192, 0, "d17ddf46adaacde531cac483de7a9367"),
            (KEY_192, 16, "9e99a7bf31e710900662f65e617c5184"),
            (KEY_192, 40, "8a1de5be2eb31aad089a82e6ee908b0e"),
            (KEY_192, 64, "a1d5df0eed790f794d77589659f39a11"),
            (KEY_256, 0, "028962f61b7bf89efc6b551f4667d983"),
            (KEY_256, 16, "28a7023f452e8f82bd4bf28d8c37c35c"),
            (KEY_256, 40, "aaf3d8f1de5640c232f5b169b9c911e6"),
            (KEY_256, 64, "e1992190549f6ed5696a2c056c315410"),
        ],
    )
    def test_official_vectors(self, key, length, tag):
        assert mac.cmac(key, CMAC_MSG[:length]).hex() == tag

    # Lengths around block boundaries: partial last block (K2) vs full last block (K1).
    # Expected tags computed with the `cryptography` library.
    @pytest.mark.parametrize(
        "length, tag",
        [
            (1, "8e48c3c1d9f1c17c295c7aefd232bb14"),
            (15, "f212d4c2154c8766de60c18c98fa0c93"),
            (17, "bc72cc168ec5a1434dcdb20bc1a2c2a4"),
            (31, "8a157acff517d21bcd6ab65cd014cc70"),
            (32, "ce0cbf1738f4df6428b1d93bf12081c9"),
            (33, "cb8006fd4b9a8313333943ad6eb92797"),
            (48, "c47c4d9d64588f67fb9de6fe745d7fbf"),
            (63, "dfd14adbe2ad17d918ed36a674afb7d7"),
        ],
    )
    def test_every_padding_case(self, length, tag):
        assert mac.cmac(KEY_128, CMAC_MSG[:length]).hex() == tag

    def test_one_full_block_by_hand(self):
        # n = 1, full block: tag = AES(K, M1 xor K1).
        k1, _ = mac.cmac_subkeys(KEY_128)
        m = CMAC_MSG[:16]
        assert mac.cmac(KEY_128, m) == aes.encrypt_block(xor(m, k1), KEY_128)

    def test_empty_by_hand(self):
        # Empty message: one padded block 80 00..00, masked with K2.
        _, k2 = mac.cmac_subkeys(KEY_128)
        padded = b"\x80" + bytes(15)
        assert mac.cmac(KEY_128, b"") == aes.encrypt_block(xor(padded, k2), KEY_128)

    def test_padding_is_not_ambiguous(self):
        # A partial message and its 10..0-padded form must get different tags (K1 vs K2).
        m = b"fifteen bytes!!"
        assert mac.cmac(KEY_128, m) != mac.cmac(KEY_128, m + b"\x80")

    def test_differs_from_cbc_mac(self):
        m = CMAC_MSG[:32]
        assert mac.cmac(KEY_128, m) != mac.cbc_mac(KEY_128, m)

    def test_returns_16_bytes(self):
        tag = mac.cmac(KEY_128, b"abc")
        assert type(tag) is bytes
        assert len(tag) == 16

    def test_accepts_bytearray(self):
        assert mac.cmac(bytearray(KEY_128), bytearray(b"abc")) == mac.cmac(KEY_128, b"abc")

    def test_does_not_modify_input(self):
        m = bytearray(CMAC_MSG[:40])
        mac.cmac(KEY_128, m)
        assert m == bytearray(CMAC_MSG[:40])


class TestCmacResistsCbcMacForgery:
    """The Step 8 forgeries no longer work: the last block is masked with a secret subkey."""

    def oracle(self, message):
        return mac.cmac(KEY_128, message)

    def test_slides_forgery_fails(self):
        m1, m2 = b"pay alice $10.00", b"pay mallory $999"
        t1 = self.oracle(m1)
        t = self.oracle(xor(t1, m2))
        assert mac.cmac(KEY_128, m1 + m2) != t

    def test_extend_with_zero_xor_fails(self):
        m = b"transfer 100 EUR"
        t = self.oracle(m)
        assert mac.cmac(KEY_128, m + xor(t, m)) != t
