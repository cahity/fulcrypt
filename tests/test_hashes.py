"""Tests for hashes.py: SHA-256 (FIPS 180-4), built step by step."""

import hashlib

import pytest

from fulcrypt import hashes

# FIPS 180-4 example messages (also in the NIST "SHA-256 examples" document)
ABC = b"abc"
MSG_448 = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"  # 56 bytes = 448 bits


# =============================================================================
# Step 3: SHA-256 padding  (FIPS 180-4 section 5.1.1)
# =============================================================================


def test_block_size():
    assert hashes.SHA256_BLOCK_SIZE == 64  # 512 bits


class TestSha256Pad:
    def test_abc(self):
        # "abc" || 0x80 || 52 zero bytes || length 24 bits as 64-bit big-endian
        expected = ABC + b"\x80" + bytes(52) + (24).to_bytes(8, "big")
        assert hashes.sha256_pad(ABC) == expected

    def test_empty(self):
        assert hashes.sha256_pad(b"") == b"\x80" + bytes(55) + bytes(8)

    def test_length_is_in_bits_not_bytes(self):
        padded = hashes.sha256_pad(ABC)
        assert padded[-8:] == (24).to_bytes(8, "big")  # 3 bytes = 24 bits, not 3

    def test_55_bytes_fits_in_one_block(self):
        # 55 + 1 (0x80) + 8 (length) = 64: no zero bytes at all
        msg = b"a" * 55
        assert hashes.sha256_pad(msg) == msg + b"\x80" + (440).to_bytes(8, "big")

    def test_56_bytes_spills_into_second_block(self):
        # 56 + 1 > 56: the length no longer fits, so a whole extra block is needed
        padded = hashes.sha256_pad(MSG_448)
        assert len(padded) == 128
        assert padded == MSG_448 + b"\x80" + bytes(63) + (448).to_bytes(8, "big")

    def test_full_block_gets_a_padding_block(self):
        msg = bytes(range(64))
        padded = hashes.sha256_pad(msg)
        assert padded == msg + b"\x80" + bytes(55) + (512).to_bytes(8, "big")

    @pytest.mark.parametrize(
        "length, padded_length",
        [(0, 64), (1, 64), (55, 64), (56, 128), (63, 128), (64, 128), (119, 128), (120, 192)],
    )
    def test_boundaries(self, length, padded_length):
        assert len(hashes.sha256_pad(bytes(length))) == padded_length

    @pytest.mark.parametrize("length", range(0, 200))
    def test_structure(self, length):
        msg = bytes((7 * i + 1) % 256 for i in range(length))
        padded = hashes.sha256_pad(msg)
        # smallest multiple of 64 that holds message + 0x80 + 8-byte length
        assert len(padded) % 64 == 0
        assert len(msg) + 9 <= len(padded) < len(msg) + 9 + 64
        assert padded[:length] == msg
        assert padded[length] == 0x80
        assert padded[length + 1 : -8] == bytes(len(padded) - length - 9)
        assert int.from_bytes(padded[-8:], "big") == 8 * length

    def test_long_message_length_field(self):
        # 1000 bytes = 8000 bits = 0x1f40: the length really is a 64-bit big-endian integer
        padded = hashes.sha256_pad(bytes(1000))
        assert padded[-8:] == bytes.fromhex("0000000000001f40")

    def test_returns_bytes(self):
        assert type(hashes.sha256_pad(bytearray(b"abc"))) is bytes

    def test_does_not_modify_input(self):
        msg = bytearray(b"abc")
        hashes.sha256_pad(msg)
        assert msg == bytearray(b"abc")


# =============================================================================
# Step 4: SHA-256 compression  (FIPS 180-4 sections 4.1.2, 4.2.2, 5.3.3, 6.2.2)
# =============================================================================


def words(hex_words):
    return tuple(int(w, 16) for w in hex_words.split())


FIPS_H0 = words("6a09e667 bb67ae85 3c6ef372 a54ff53a 510e527f 9b05688c 1f83d9ab 5be0cd19")
FIPS_K = words(
    "428a2f98 71374491 b5c0fbcf e9b5dba5 3956c25b 59f111f1 923f82a4 ab1c5ed5 "
    "d807aa98 12835b01 243185be 550c7dc3 72be5d74 80deb1fe 9bdc06a7 c19bf174 "
    "e49b69c1 efbe4786 0fc19dc6 240ca1cc 2de92c6f 4a7484aa 5cb0a9dc 76f988da "
    "983e5152 a831c66d b00327c8 bf597fc7 c6e00bf3 d5a79147 06ca6351 14292967 "
    "27b70a85 2e1b2138 4d2c6dfc 53380d13 650a7354 766a0abb 81c2c92e 92722c85 "
    "a2bfe8a1 a81a664b c24b8b70 c76c51a3 d192e819 d6990624 f40e3585 106aa070 "
    "19a4c116 1e376c08 2748774c 34b0bcb5 391c0cb3 4ed8aa4a 5b9cca4f 682e6ff3 "
    "748f82ee 78a5636f 84c87814 8cc70208 90befffa a4506ceb bef9a3f7 c67178f2"
)


class TestSha256Constants:
    def test_h0(self):
        assert hashes.SHA256_H0 == FIPS_H0

    def test_k(self):
        assert hashes.SHA256_K == FIPS_K

    def test_are_tuples(self):
        assert type(hashes.SHA256_H0) is tuple and len(hashes.SHA256_H0) == 8
        assert type(hashes.SHA256_K) is tuple and len(hashes.SHA256_K) == 64


# FIPS 180-4 section 5.3.2: second 32 bits of the fractional parts of sqrt of primes 9..16
FIPS_H0_224 = words("c1059ed8 367cd507 3070dd17 f70e5939 ffc00b31 68581511 64f98fa7 befa4fa4")


class TestSha224Constants:
    def test_h0(self):
        assert hashes.SHA224_H0 == FIPS_H0_224

    def test_first_word(self):
        # prime 23 (the 9th prime) -> c1059ed8
        assert hashes.SHA224_H0[0] == 0xC1059ED8

    def test_last_word(self):
        # prime 53 (the 16th prime) -> befa4fa4
        assert hashes.SHA224_H0[-1] == 0xBEFA4FA4

    def test_is_tuple(self):
        assert type(hashes.SHA224_H0) is tuple and len(hashes.SHA224_H0) == 8

    def test_all_32_bit(self):
        assert all(0 <= w <= hashes.MASK32 for w in hashes.SHA224_H0)


class TestSha256Compress:
    def test_abc_single_block(self):
        # One block from H0: the result is the final digest of "abc" (FIPS 180-4 example).
        state = hashes.sha256_compress(FIPS_H0, hashes.sha256_pad(ABC))
        assert state == words(
            "ba7816bf 8f01cfea 414140de 5dae2223 b00361a3 96177a9c b410ff61 f20015ad"
        )

    def test_empty_message(self):
        state = hashes.sha256_compress(FIPS_H0, hashes.sha256_pad(b""))
        assert state == words(
            "e3b0c442 98fc1c14 9afbf4c8 996fb924 27ae41e4 649b934c a495991b 7852b855"
        )

    def test_two_blocks_intermediate_and_final(self):
        # FIPS 180-4 448-bit example: H(1) after the first block, then the digest.
        padded = hashes.sha256_pad(MSG_448)
        h1 = hashes.sha256_compress(FIPS_H0, padded[:64])
        assert h1 == words(
            "85e655d6 417a1795 3363376a 624cde5c 76e09589 cac5f811 cc4b32c1 f20e533a"
        )
        h2 = hashes.sha256_compress(h1, padded[64:])
        assert h2 == words(
            "248d6a61 d20638b8 e5c02693 0c3e6039 a33ce459 64ff2167 f6ecedd4 19db06c1"
        )

    def test_zero_state_zero_block(self):
        # No message or H0 involved: checks the rounds and the feed-forward on their own.
        state = hashes.sha256_compress((0,) * 8, bytes(64))
        assert state == words(
            "7ca51614 425c3ba8 ce54dd2f c2020ae7 b6e574d1 98136d0f ae7e26cc bf0be7a6"
        )

    @pytest.mark.parametrize("length", [0, 3, 55, 56, 64, 100, 200, 1000])
    def test_chain_matches_hashlib(self, length):
        # Merkle-Damgard by hand: chaining your compression over the padded message gives SHA-256.
        msg = bytes((11 * i + 5) % 256 for i in range(length))
        padded = hashes.sha256_pad(msg)
        state = FIPS_H0
        for i in range(0, len(padded), 64):
            state = hashes.sha256_compress(state, padded[i : i + 64])
        digest = b"".join(w.to_bytes(4, "big") for w in state)
        assert digest == hashlib.sha256(msg).digest()

    def test_output_shape(self):
        state = hashes.sha256_compress(FIPS_H0, bytes(64))
        assert type(state) is tuple and len(state) == 8
        assert all(type(w) is int and 0 <= w <= 0xFFFFFFFF for w in state)

    def test_input_state_unchanged(self):
        state = list(FIPS_H0)
        hashes.sha256_compress(state, bytes(64))
        assert state == list(FIPS_H0)

    def test_accepts_bytearray_block(self):
        a = hashes.sha256_compress(FIPS_H0, bytearray(64))
        assert a == hashes.sha256_compress(FIPS_H0, bytes(64))

    def test_one_bit_changes_about_half_the_output(self):
        # Avalanche: flipping one input bit flips ~128 of the 256 output bits.
        base = hashes.sha256_compress(FIPS_H0, bytes(64))
        total = 0
        for bit in range(0, 512, 16):
            block = bytearray(64)
            block[bit // 8] ^= 0x80 >> (bit % 8)
            out = hashes.sha256_compress(FIPS_H0, bytes(block))
            total += sum(bin(x ^ y).count("1") for x, y in zip(base, out, strict=True))
        assert 108 < total / 32 < 148

    @pytest.mark.parametrize("length", [0, 63, 65, 128])
    def test_rejects_bad_block_length(self, length):
        with pytest.raises(ValueError):
            hashes.sha256_compress(FIPS_H0, bytes(length))

    @pytest.mark.parametrize("n", [0, 7, 9])
    def test_rejects_bad_state_length(self, n):
        with pytest.raises(ValueError):
            hashes.sha256_compress((0,) * n, bytes(64))


# =============================================================================
# Step 5: SHA-256  (FIPS 180-4 section 6.2)
# =============================================================================

MSG_896 = (
    b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
    b"hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
)  # 112 bytes = 896 bits


def test_digest_size():
    assert hashes.SHA256_DIGEST_SIZE == 32


class TestSha256:
    @pytest.mark.parametrize(
        "message, digest",
        [
            (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            (ABC, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            (MSG_448, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
            (MSG_896, "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"),
        ],
    )
    def test_fips_examples(self, message, digest):
        assert hashes.sha256(message).hex() == digest

    @pytest.mark.timeout(10)  # ~15,600 compressions in pure Python: slower than the 1 s default
    def test_one_million_a(self):
        expected = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        assert hashes.sha256(b"a" * 1_000_000).hex() == expected

    @pytest.mark.parametrize("length", range(0, 260))
    def test_matches_hashlib_every_length(self, length):
        # Covers every padding case: 0-55 (one block), 56-63 (spill), 64+ (multi-block).
        msg = bytes((13 * i + 7) % 256 for i in range(length))
        assert hashes.sha256(msg) == hashlib.sha256(msg).digest()

    def test_returns_32_bytes(self):
        digest = hashes.sha256(b"abc")
        assert type(digest) is bytes
        assert len(digest) == 32

    def test_accepts_bytearray(self):
        assert hashes.sha256(bytearray(b"abc")) == hashes.sha256(b"abc")

    def test_does_not_modify_input(self):
        msg = bytearray(b"abc")
        hashes.sha256(msg)
        assert msg == bytearray(b"abc")

    def test_is_the_compression_chain(self):
        # sha256 = pad, then compress block by block from H0, output big-endian.
        padded = hashes.sha256_pad(MSG_896)
        state = hashes.SHA256_H0
        for i in range(0, len(padded), 64):
            state = hashes.sha256_compress(state, padded[i : i + 64])
        assert hashes.sha256(MSG_896) == b"".join(w.to_bytes(4, "big") for w in state)

    def test_does_not_change_h0(self):
        hashes.sha256(b"abc")
        assert hashes.SHA256_H0 == FIPS_H0

    def test_avalanche(self):
        # Messages differing in one bit give digests differing in ~half (128) of their bits.
        base = hashes.sha256(b"The quick brown fox jumps over the lazy dog")
        total = 0
        for i in range(43):
            msg = bytearray(b"The quick brown fox jumps over the lazy dog")
            msg[i] ^= 0x01
            other = hashes.sha256(bytes(msg))
            total += sum(bin(x ^ y).count("1") for x, y in zip(base, other, strict=True))
        assert 108 < total / 43 < 148


class TestSha224:
    @pytest.mark.parametrize(
        "message, digest",
        [
            (b"", "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f"),
            (ABC, "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7"),
            (MSG_448, "75388b16512776cc5dba5da1fd890150b0c6455cb4f58b1952522525"),
            (MSG_896, "c97ca9a559850ce97a04a96def6d99a9e0e0e2ab14e6b8df265fc0b3"),
        ],
    )
    def test_fips_examples(self, message, digest):
        assert hashes.sha224(message).hex() == digest

    @pytest.mark.parametrize("length", [0, 1, 55, 56, 63, 64, 65, 119, 120, 128, 200])
    def test_matches_hashlib(self, length):
        msg = bytes((13 * i + 7) % 256 for i in range(length))
        assert hashes.sha224(msg) == hashlib.sha224(msg).digest()

    def test_returns_28_bytes(self):
        digest = hashes.sha224(b"abc")
        assert type(digest) is bytes
        assert len(digest) == 28

    def test_is_not_truncated_sha256(self):
        # Different initial state: not just the first 28 bytes of SHA-256.
        assert hashes.sha224(b"abc") != hashes.sha256(b"abc")[:28]

    def test_is_first_seven_words_of_chain(self):
        padded = hashes.sha256_pad(MSG_896)
        state = hashes.SHA224_H0
        for i in range(0, len(padded), 64):
            state = hashes.sha256_compress(state, padded[i : i + 64])
        assert hashes.sha224(MSG_896) == b"".join(w.to_bytes(4, "big") for w in state[:7])

    def test_does_not_change_h0(self):
        hashes.sha224(b"abc")
        assert hashes.SHA224_H0 == FIPS_H0_224


# =============================================================================
# Step 6: Length extension attack  (slides 4a p28-29)
# =============================================================================

SECRET = b"server-side secret key"  # 22 bytes; the attacker never sees it
QUERY = b"user=alice&role=user"


def glue_for(length):
    # SHA-256 padding of a message of the given length (padding depends only on the length).
    return hashes.sha256_pad(b"\x00" * length)[length:]


class TestSha256Extend:
    def test_returns_bytes_pair(self):
        ext, digest = hashes.sha256_extend(hashes.sha256(b"abc"), 3, b"x")
        assert type(ext) is bytes and type(digest) is bytes
        assert len(digest) == 32

    def test_extension_is_glue_then_suffix(self):
        ext, _ = hashes.sha256_extend(hashes.sha256(b"abc"), 3, b"&admin=true")
        assert ext == glue_for(3) + b"&admin=true"

    def test_glue_abc(self):
        # "abc" is 24 bits: 0x80, 52 zero bytes, then 0x18 as a 64-bit length.
        ext, _ = hashes.sha256_extend(hashes.sha256(b"abc"), 3, b"")
        assert ext == b"\x80" + b"\x00" * 52 + (24).to_bytes(8, "big")

    def test_forged_mac(self):
        # The server's MAC is sha256(secret || query); the attacker knows only the MAC
        # and the total length, and forges a valid MAC for a longer query.
        mac = hashes.sha256(SECRET + QUERY)
        ext, forged = hashes.sha256_extend(mac, len(SECRET) + len(QUERY), b"&role=admin")
        assert forged == hashes.sha256(SECRET + QUERY + ext)
        assert forged == hashlib.sha256(SECRET + QUERY + ext).digest()

    # Original lengths around the padding boundaries: one block, spill, exact block, multi-block.
    @pytest.mark.parametrize("original_length", [0, 1, 54, 55, 56, 63, 64, 65, 119, 120, 128, 200])
    def test_any_original_length(self, original_length):
        original = bytes((7 * i + 3) % 256 for i in range(original_length))
        suffix = b"&admin=true"
        ext, forged = hashes.sha256_extend(hashes.sha256(original), original_length, suffix)
        assert forged == hashlib.sha256(original + ext).digest()

    # Suffix lengths that make the forged message's own padding fit or spill.
    @pytest.mark.parametrize("suffix_length", [0, 1, 55, 56, 63, 64, 65, 130])
    def test_any_suffix_length(self, suffix_length):
        original = SECRET + QUERY
        suffix = bytes((5 * i + 1) % 256 for i in range(suffix_length))
        ext, forged = hashes.sha256_extend(hashes.sha256(original), len(original), suffix)
        assert ext.endswith(suffix)
        assert forged == hashlib.sha256(original + ext).digest()

    def test_empty_suffix(self):
        # Still a valid forgery: the original message plus its own padding.
        ext, forged = hashes.sha256_extend(hashes.sha256(b"abc"), 3, b"")
        assert ext == glue_for(3)
        assert forged == hashlib.sha256(b"abc" + glue_for(3)).digest()

    def test_extend_twice(self):
        # A forged digest can itself be extended.
        original = SECRET + QUERY
        ext1, d1 = hashes.sha256_extend(hashes.sha256(original), len(original), b"&a=1")
        ext2, d2 = hashes.sha256_extend(d1, len(original) + len(ext1), b"&b=2")
        assert d2 == hashlib.sha256(original + ext1 + ext2).digest()

    def test_wrong_length_guess_fails(self):
        # The attack needs the exact length of secret || m.
        original = SECRET + QUERY
        ext, forged = hashes.sha256_extend(hashes.sha256(original), len(original) + 1, b"&x")
        assert forged != hashlib.sha256(original + ext).digest()

    def test_accepts_bytearray(self):
        mac = hashes.sha256(b"abc")
        assert hashes.sha256_extend(bytearray(mac), 3, bytearray(b"x")) == hashes.sha256_extend(
            mac, 3, b"x"
        )

    @pytest.mark.parametrize("bad", [b"", b"\x00" * 31, b"\x00" * 33, b"\x00" * 28])
    def test_rejects_wrong_digest_size(self, bad):
        with pytest.raises(ValueError):
            hashes.sha256_extend(bad, 3, b"x")

    def test_rejects_negative_length(self):
        with pytest.raises(ValueError):
            hashes.sha256_extend(hashes.sha256(b"abc"), -1, b"x")
