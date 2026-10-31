import hashlib
import hmac

import pytest

from fulcrypt import kdf, mac

# =============================================================================
# Step 10: HKDF-SHA256  (RFC 5869, test vectors Appendix A.1-A.3)
# =============================================================================

RFC5869 = [
    pytest.param(  # A.1: basic case
        b"\x0b" * 22,
        bytes(range(0x00, 0x0D)),
        bytes(range(0xF0, 0xFA)),
        42,
        "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5",
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865",
        id="A.1",
    ),
    pytest.param(  # A.2: longer inputs and output (3 HMAC blocks)
        bytes(range(0x00, 0x50)),
        bytes(range(0x60, 0xB0)),
        bytes(range(0xB0, 0x100)),
        82,
        "06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244",
        "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c"
        "59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71"
        "cc30c58179ec3e87c14c01d5c1f3434f1d87",
        id="A.2",
    ),
    pytest.param(  # A.3: empty salt and info
        b"\x0b" * 22,
        b"",
        b"",
        42,
        "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04",
        "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8",
        id="A.3",
    ),
]


def ref_expand(prk, info, length):
    # RFC 5869 section 2.3 with the stdlib hmac, as an independent reference.
    out, t, i = b"", b"", 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:length]


class TestHkdfExtract:
    @pytest.mark.parametrize("ikm, salt, info, length, prk, okm", RFC5869)
    def test_rfc5869(self, ikm, salt, info, length, prk, okm):
        assert kdf.hkdf_extract(salt, ikm).hex() == prk

    def test_is_hmac_with_salt_as_key(self):
        assert kdf.hkdf_extract(b"salt", b"ikm") == mac.hmac_sha256(b"salt", b"ikm")

    def test_empty_salt_is_32_zero_bytes(self):
        assert kdf.hkdf_extract(b"", b"ikm") == kdf.hkdf_extract(bytes(32), b"ikm")

    def test_returns_32_bytes(self):
        prk = kdf.hkdf_extract(b"salt", b"ikm")
        assert type(prk) is bytes
        assert len(prk) == 32


class TestHkdfExpand:
    @pytest.mark.parametrize("ikm, salt, info, length, prk, okm", RFC5869)
    def test_rfc5869(self, ikm, salt, info, length, prk, okm):
        assert kdf.hkdf_expand(bytes.fromhex(prk), info, length).hex() == okm

    # Around the 32-byte HMAC output: part of a block, one block, a block and a bit, many.
    @pytest.mark.parametrize("length", [1, 31, 32, 33, 64, 65, 100, 255 * 32])
    def test_matches_reference_every_length(self, length):
        prk = bytes(range(32))
        assert kdf.hkdf_expand(prk, b"info", length) == ref_expand(prk, b"info", length)

    def test_first_block(self):
        # T(1) = HMAC(PRK, info || 0x01): the empty T(0) contributes nothing.
        prk = bytes(range(32))
        assert kdf.hkdf_expand(prk, b"ctx", 32) == mac.hmac_sha256(prk, b"ctx" + b"\x01")

    def test_second_block_chains_the_first(self):
        # T(2) = HMAC(PRK, T(1) || info || 0x02).
        prk = bytes(range(32))
        okm = kdf.hkdf_expand(prk, b"ctx", 64)
        assert okm[32:] == mac.hmac_sha256(prk, okm[:32] + b"ctx" + b"\x02")

    def test_shorter_output_is_a_prefix(self):
        prk = bytes(range(32))
        long = kdf.hkdf_expand(prk, b"info", 100)
        for length in (0, 1, 32, 50, 99):
            assert kdf.hkdf_expand(prk, b"info", length) == long[:length]

    def test_zero_length(self):
        assert kdf.hkdf_expand(bytes(32), b"info", 0) == b""

    def test_different_info_independent_keys(self):
        # The slides' k_i <- f(k, x_i): one master key, many keys by changing info.
        prk = bytes(range(32))
        enc = kdf.hkdf_expand(prk, b"encryption", 32)
        auth = kdf.hkdf_expand(prk, b"authentication", 32)
        assert enc != auth

    def test_maximum_length(self):
        # The counter is one byte, so at most 255 blocks of 32 bytes.
        assert len(kdf.hkdf_expand(bytes(32), b"", 255 * 32)) == 8160

    @pytest.mark.parametrize("length", [255 * 32 + 1, 10_000, -1])
    def test_rejects_bad_length(self, length):
        with pytest.raises(ValueError):
            kdf.hkdf_expand(bytes(32), b"", length)


class TestHkdf:
    @pytest.mark.parametrize("ikm, salt, info, length, prk, okm", RFC5869)
    def test_rfc5869(self, ikm, salt, info, length, prk, okm):
        assert kdf.hkdf(ikm, salt, info, length).hex() == okm

    def test_is_extract_then_expand(self):
        prk = kdf.hkdf_extract(b"salt", b"ikm")
        assert kdf.hkdf(b"ikm", b"salt", b"info", 50) == kdf.hkdf_expand(prk, b"info", 50)

    def test_salt_matters(self):
        assert kdf.hkdf(b"ikm", b"salt1", b"", 32) != kdf.hkdf(b"ikm", b"salt2", b"", 32)

    def test_accepts_bytearray(self):
        args = (b"ikm", b"salt", b"info", 40)
        assert kdf.hkdf(*(bytearray(a) for a in args[:3]), 40) == kdf.hkdf(*args)

    def test_rejects_bad_length(self):
        with pytest.raises(ValueError):
            kdf.hkdf(b"ikm", b"salt", b"info", 255 * 32 + 1)


# =============================================================================
# Step 11: PBKDF2-HMAC-SHA256  (RFC 8018 section 5.2, test vector RFC 7914 section 11)
# =============================================================================
# Our HMAC takes ~0.4 ms, so 1000 iterations cost ~0.4 s per 32-byte block. The official
# vectors with 80,000+ iterations would take minutes; small counts against hashlib instead.


def int_be(i):
    return i.to_bytes(4, "big")


class TestPbkdf2Sha256:
    def test_rfc7914(self):
        expected = (
            "55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc"
            "49ca9cccf179b645991664b39d77ef317c71b845b1e30bd509112041d3a19783"
        )
        assert kdf.pbkdf2_sha256(b"passwd", b"salt", 1, 64).hex() == expected

    @pytest.mark.parametrize("iterations", [1, 2, 3, 10, 100])
    def test_matches_hashlib_iterations(self, iterations):
        expected = hashlib.pbkdf2_hmac("sha256", b"password", b"salt", iterations, 32)
        assert kdf.pbkdf2_sha256(b"password", b"salt", iterations, 32) == expected

    # Part of a block, one block, a block and a bit, several blocks.
    @pytest.mark.parametrize("length", [1, 20, 31, 32, 33, 64, 65, 100])
    def test_matches_hashlib_lengths(self, length):
        expected = hashlib.pbkdf2_hmac("sha256", b"password", b"salt", 2, length)
        assert kdf.pbkdf2_sha256(b"password", b"salt", 2, length) == expected

    @pytest.mark.timeout(5)  # 1000 HMACs in pure Python: ~0.4 s, more on a slow machine
    def test_thousand_iterations(self):
        expected = hashlib.pbkdf2_hmac("sha256", b"correct horse", b"NaCl", 1000, 32)
        assert kdf.pbkdf2_sha256(b"correct horse", b"NaCl", 1000, 32) == expected

    def test_one_iteration_is_one_hmac(self):
        # U1 = HMAC(P, S || INT(1)), with INT(i) a 4-byte big-endian block index.
        assert kdf.pbkdf2_sha256(b"pw", b"salt", 1, 32) == mac.hmac_sha256(
            b"pw", b"salt" + int_be(1)
        )

    def test_iterations_are_xored(self):
        # T1 = U1 xor U2 xor U3, with U2 = HMAC(P, U1) and U3 = HMAC(P, U2).
        u1 = mac.hmac_sha256(b"pw", b"salt" + int_be(1))
        u2 = mac.hmac_sha256(b"pw", u1)
        u3 = mac.hmac_sha256(b"pw", u2)
        expected = bytes(a ^ b ^ c for a, b, c in zip(u1, u2, u3, strict=True))
        assert kdf.pbkdf2_sha256(b"pw", b"salt", 3, 32) == expected

    def test_second_block_uses_index_2(self):
        # Every output block restarts from the salt with its own index; blocks don't chain.
        out = kdf.pbkdf2_sha256(b"pw", b"salt", 1, 64)
        assert out[32:] == mac.hmac_sha256(b"pw", b"salt" + int_be(2))

    def test_shorter_output_is_a_prefix(self):
        long = kdf.pbkdf2_sha256(b"pw", b"salt", 2, 80)
        for length in (0, 1, 32, 50, 79):
            assert kdf.pbkdf2_sha256(b"pw", b"salt", 2, length) == long[:length]

    def test_zero_length(self):
        assert kdf.pbkdf2_sha256(b"pw", b"salt", 1, 0) == b""

    def test_empty_password_and_salt(self):
        expected = hashlib.pbkdf2_hmac("sha256", b"", b"", 2, 32)
        assert kdf.pbkdf2_sha256(b"", b"", 2, 32) == expected

    def test_long_password(self):
        # Longer than the HMAC block: hmac_sha256 hashes it first.
        pw = b"p" * 100
        assert kdf.pbkdf2_sha256(pw, b"salt", 2, 32) == hashlib.pbkdf2_hmac(
            "sha256", pw, b"salt", 2, 32
        )

    def test_every_input_matters(self):
        base = kdf.pbkdf2_sha256(b"pw", b"salt", 2, 32)
        assert kdf.pbkdf2_sha256(b"pW", b"salt", 2, 32) != base
        assert kdf.pbkdf2_sha256(b"pw", b"salT", 2, 32) != base
        assert kdf.pbkdf2_sha256(b"pw", b"salt", 3, 32) != base

    def test_returns_bytes(self):
        out = kdf.pbkdf2_sha256(b"pw", b"salt", 1, 40)
        assert type(out) is bytes
        assert len(out) == 40

    def test_accepts_bytearray(self):
        assert kdf.pbkdf2_sha256(bytearray(b"pw"), bytearray(b"salt"), 2, 32) == (
            kdf.pbkdf2_sha256(b"pw", b"salt", 2, 32)
        )

    @pytest.mark.parametrize("iterations", [0, -1])
    def test_rejects_bad_iterations(self, iterations):
        with pytest.raises(ValueError):
            kdf.pbkdf2_sha256(b"pw", b"salt", iterations, 32)

    def test_rejects_negative_length(self):
        with pytest.raises(ValueError):
            kdf.pbkdf2_sha256(b"pw", b"salt", 1, -1)
