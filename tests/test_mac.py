import hashlib
import hmac

import pytest

from fulcrypt import hashes, mac

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
