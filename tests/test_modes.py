"""Tests for modes.py: padding, ECB and CBC. Built step by step.

Design: modes work on plain bytes; mode functions take block-aligned data,
and padding is a separate layer.
"""

import os
import random

import pytest

from fulcrypt import modes


def random_bytes(rng, n):
    return bytes(rng.randrange(256) for _ in range(n))


# =============================================================================
# Step 0 / 1: constants and byte helpers
# =============================================================================


def test_block_size():
    assert modes.BLOCK_SIZE == 16


class TestXorBytes:
    def test_known(self):
        assert modes.xor_bytes(b"\x0f\xf0\xaa", b"\xff\xff\x55") == b"\xf0\x0f\xff"

    def test_returns_bytes(self):
        assert type(modes.xor_bytes(b"ab", b"cd")) is bytes

    def test_empty(self):
        assert modes.xor_bytes(b"", b"") == b""

    def test_self_is_zero(self):
        data = bytes(range(32))
        assert modes.xor_bytes(data, data) == bytes(32)

    def test_zero_is_identity(self):
        data = bytes(range(32))
        assert modes.xor_bytes(data, bytes(32)) == data

    def test_commutative_and_self_inverse(self):
        rng = random.Random(1)
        for _ in range(50):
            n = rng.randrange(1, 40)
            a, b = random_bytes(rng, n), random_bytes(rng, n)
            assert modes.xor_bytes(a, b) == modes.xor_bytes(b, a)
            assert modes.xor_bytes(modes.xor_bytes(a, b), b) == a

    def test_accepts_bytearray(self):
        assert modes.xor_bytes(bytearray(b"\x01\x02"), b"\x03\x03") == b"\x02\x01"

    @pytest.mark.parametrize("la, lb", [(1, 0), (0, 1), (15, 16), (16, 15), (16, 32)])
    def test_rejects_different_lengths(self, la, lb):
        # zip() would silently stop at the shorter input and hide the bug.
        with pytest.raises(ValueError):
            modes.xor_bytes(bytes(la), bytes(lb))


class TestSplitBlocks:
    def test_empty(self):
        assert modes.split_blocks(b"") == []

    def test_one_block(self):
        data = bytes(range(16))
        assert modes.split_blocks(data) == [data]

    def test_several_blocks_in_order(self):
        data = bytes(range(48))
        assert modes.split_blocks(data) == [
            bytes(range(0, 16)),
            bytes(range(16, 32)),
            bytes(range(32, 48)),
        ]

    def test_blocks_are_bytes_of_block_size(self):
        blocks = modes.split_blocks(bytes(64))
        assert len(blocks) == 4
        assert all(type(b) is bytes and len(b) == 16 for b in blocks)

    def test_join_round_trip(self):
        rng = random.Random(2)
        for n in range(0, 129, 16):
            data = random_bytes(rng, n)
            assert b"".join(modes.split_blocks(data)) == data

    @pytest.mark.parametrize("block_size, count", [(1, 24), (4, 6), (8, 3), (24, 1)])
    def test_custom_block_size(self, block_size, count):
        data = bytes(range(24))
        blocks = modes.split_blocks(data, block_size)
        assert len(blocks) == count
        assert all(len(b) == block_size for b in blocks)
        assert b"".join(blocks) == data

    @pytest.mark.parametrize("length", [1, 15, 17, 31, 33])
    def test_rejects_unaligned(self, length):
        with pytest.raises(ValueError):
            modes.split_blocks(bytes(length))

    def test_rejects_unaligned_custom_block_size(self):
        with pytest.raises(ValueError):
            modes.split_blocks(bytes(10), 8)


class TestSplitBlocksUnaligned:
    def test_empty(self):
        assert modes.split_blocks_unaligned(b"") == []

    def test_partial_last_block(self):
        data = bytes(range(40))
        assert modes.split_blocks_unaligned(data) == [
            bytes(range(0, 16)),
            bytes(range(16, 32)),
            bytes(range(32, 40)),
        ]

    def test_shorter_than_one_block(self):
        assert modes.split_blocks_unaligned(b"abc") == [b"abc"]

    @pytest.mark.parametrize("length", [16, 32, 64])
    def test_aligned_same_as_split_blocks(self, length):
        data = bytes(range(length))
        assert modes.split_blocks_unaligned(data) == modes.split_blocks(data)

    @pytest.mark.parametrize("length", range(0, 50))
    def test_shape(self, length):
        # ceil(length / 16) blocks; all full except possibly the last; joining restores the data.
        data = bytes((3 * i) % 256 for i in range(length))
        blocks = modes.split_blocks_unaligned(data)
        assert len(blocks) == -(-length // 16)
        assert all(len(b) == 16 for b in blocks[:-1])
        if blocks:
            assert 1 <= len(blocks[-1]) <= 16
        assert b"".join(blocks) == data

    @pytest.mark.parametrize(
        "block_size, sizes", [(1, [1] * 10), (3, [3, 3, 3, 1]), (4, [4, 4, 2])]
    )
    def test_custom_block_size(self, block_size, sizes):
        blocks = modes.split_blocks_unaligned(bytes(10), block_size)
        assert [len(b) for b in blocks] == sizes

    def test_returns_bytes(self):
        blocks = modes.split_blocks_unaligned(bytearray(20))
        assert all(type(b) is bytes for b in blocks)

    @pytest.mark.parametrize("block_size", [0, -1])
    def test_rejects_bad_block_size(self, block_size):
        with pytest.raises(ValueError):
            modes.split_blocks_unaligned(b"abc", block_size)


# =============================================================================
# Step 2: PKCS#7 padding  (RFC 5652 section 6.3)
# =============================================================================


class TestPad:
    @pytest.mark.parametrize(
        "data, padded",
        [
            (b"", b"\x10" * 16),  # empty -> one full block of padding
            (b"abc", b"abc" + b"\x0d" * 13),
            (b"A" * 15, b"A" * 15 + b"\x01"),
            (
                b"YELLOW SUBMARINE",
                b"YELLOW SUBMARINE" + b"\x10" * 16,
            ),  # aligned -> full extra block
            (b"B" * 17, b"B" * 17 + b"\x0f" * 15),
        ],
    )
    def test_known(self, data, padded):
        assert modes.pad(data) == padded

    @pytest.mark.parametrize(
        "data, block_size, padded",
        [
            (b"abc", 8, b"abc" + b"\x05" * 5),
            (b"YELLOW SUBMARINE", 20, b"YELLOW SUBMARINE" + b"\x04" * 4),
            (b"", 4, b"\x04" * 4),
            (b"xyz", 1, b"xyz\x01"),
        ],
    )
    def test_custom_block_size(self, data, block_size, padded):
        assert modes.pad(data, block_size) == padded

    @pytest.mark.parametrize("length", range(65))
    def test_length_and_bytes(self, length):
        data = bytes(range(length))
        out = modes.pad(data)
        n = 16 - length % 16
        assert len(out) == length + n
        assert len(out) % 16 == 0
        assert out[:length] == data
        assert out[length:] == bytes([n]) * n

    def test_returns_bytes(self):
        assert type(modes.pad(b"abc")) is bytes
        assert type(modes.pad(bytearray(b"abc"))) is bytes

    def test_does_not_modify_input(self):
        data = bytearray(b"abc")
        modes.pad(data)
        assert data == bytearray(b"abc")


class TestUnpad:
    @pytest.mark.parametrize(
        "padded, data",
        [
            (b"\x10" * 16, b""),
            (b"abc" + b"\x0d" * 13, b"abc"),
            (b"A" * 15 + b"\x01", b"A" * 15),
            (b"YELLOW SUBMARINE" + b"\x10" * 16, b"YELLOW SUBMARINE"),
        ],
    )
    def test_known(self, padded, data):
        assert modes.unpad(padded) == data

    @pytest.mark.parametrize("length", range(65))
    @pytest.mark.parametrize("block_size", [1, 8, 16, 20])
    def test_round_trip(self, length, block_size):
        data = bytes((7 * i) % 256 for i in range(length))
        assert modes.unpad(modes.pad(data, block_size), block_size) == data

    def test_data_that_looks_like_padding(self):
        # Messages ending in bytes like \x01 or \x02\x02 must survive:
        # pad always adds at least one byte.
        for data in (b"\x01", b"hello\x01", b"hello\x02\x02", b"\x10" * 16):
            assert modes.unpad(modes.pad(data)) == data

    def test_returns_bytes(self):
        assert type(modes.unpad(b"abc" + b"\x0d" * 13)) is bytes
        assert type(modes.unpad(bytearray(b"abc" + b"\x0d" * 13))) is bytes

    @pytest.mark.parametrize(
        "bad",
        [
            b"",  # nothing to unpad
            b"A" * 15,  # not a multiple of the block size
            b"A" * 17,
            b"A" * 15 + b"\x00",  # n = 0 is never valid
            b"A" * 15 + b"\x11",  # n = 17 > block size
            b"A" * 31 + b"\xff",  # n = 255
            b"A" * 14 + b"\x01\x02",  # n = 2, but the byte before is 0x01
            b"A" * 13 + b"\x02\x03\x03",  # n = 3, first padding byte wrong
            b"\x10" * 15 + b"\x0f",  # n = 15, but all 15 must be 0x0f
            b"\x0f" + b"\x10" * 15,  # n = 16, but one byte is 0x0f
        ],
    )
    def test_rejects_invalid_padding(self, bad):
        with pytest.raises(ValueError):
            modes.unpad(bad)

    def test_rejects_n_larger_than_custom_block_size(self):
        with pytest.raises(ValueError):
            modes.unpad(b"A" * 7 + b"\x09", 8)

    def test_checks_every_padding_byte(self):
        # Change each padding byte in turn: every single change must be caught.
        good = b"abc" + b"\x0d" * 13
        assert modes.unpad(good) == b"abc"
        for i in range(3, 16):
            bad = bytearray(good)
            bad[i] ^= 0x01
            with pytest.raises(ValueError):
                modes.unpad(bytes(bad))


# =============================================================================
# Step 3: ECB  (NIST SP 800-38A section 6.1, Appendix F.1)
# =============================================================================

# SP 800-38A Appendix F: the same four plaintext blocks are used for every mode.
SP800_PLAINTEXT = bytes.fromhex(
    "6bc1bee22e409f96e93d7e117393172a"
    "ae2d8a571e03ac9c9eb76fac45af8e51"
    "30c81c46a35ce411e5fbc1191a0a52ef"
    "f69f2445df4f9b17ad2b417be66c3710"
)
KEY_128 = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
KEY_256 = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")


class TestECB:
    # F.1.1 / F.1.2: ECB-AES128
    CT_128 = bytes.fromhex(
        "3ad77bb40d7a3660a89ecaf32466ef97"
        "f5d3d58503b9699de785895a96fdbaaf"
        "43b1cd7f598ece23881b00e3ed030688"
        "7b0c785e27e8ad3f8223207104725dd4"
    )
    # F.1.5 / F.1.6: ECB-AES256
    CT_256 = bytes.fromhex(
        "f3eed1bdb5d2a03c064b5a7e3db181f8"
        "591ccb10d410ed26dc5ba74a31362870"
        "b6ed21b99ca6f4f9f153e7b1beafed1d"
        "23304b7a39f9f3ff067d8d8f9e24ecc7"
    )

    def test_encrypt_aes128(self):
        assert modes.ecb_encrypt(SP800_PLAINTEXT, KEY_128) == self.CT_128

    def test_decrypt_aes128(self):
        assert modes.ecb_decrypt(self.CT_128, KEY_128) == SP800_PLAINTEXT

    def test_encrypt_aes256(self):
        assert modes.ecb_encrypt(SP800_PLAINTEXT, KEY_256) == self.CT_256

    def test_decrypt_aes256(self):
        assert modes.ecb_decrypt(self.CT_256, KEY_256) == SP800_PLAINTEXT

    def test_single_block_is_plain_aes(self):
        from fulcrypt import aes

        block = SP800_PLAINTEXT[:16]
        assert modes.ecb_encrypt(block, KEY_128) == aes.encrypt_block(block, KEY_128)

    def test_empty(self):
        assert modes.ecb_encrypt(b"", KEY_128) == b""
        assert modes.ecb_decrypt(b"", KEY_128) == b""

    def test_does_not_pad(self):
        # Aligned input keeps its length: padding is the caller's job.
        assert len(modes.ecb_encrypt(bytes(32), KEY_128)) == 32

    def test_returns_bytes(self):
        assert type(modes.ecb_encrypt(bytearray(SP800_PLAINTEXT), KEY_128)) is bytes
        assert type(modes.ecb_decrypt(bytearray(self.CT_128), KEY_128)) is bytes

    @pytest.mark.parametrize("length", [1, 15, 17, 33])
    def test_rejects_unaligned(self, length):
        with pytest.raises(ValueError):
            modes.ecb_encrypt(bytes(length), KEY_128)
        with pytest.raises(ValueError):
            modes.ecb_decrypt(bytes(length), KEY_128)

    def test_rejects_bad_key(self):
        with pytest.raises(ValueError):
            modes.ecb_encrypt(bytes(16), bytes(20))

    def test_blocks_are_independent(self):
        # Each block depends only on its own plaintext: block 2 is the same wherever it appears.
        ct = modes.ecb_encrypt(SP800_PLAINTEXT, KEY_128)
        alone = modes.ecb_encrypt(SP800_PLAINTEXT[32:48], KEY_128)
        assert ct[32:48] == alone

    def test_swapping_ciphertext_blocks_swaps_plaintext(self):
        # ECB weakness: an attacker can reorder blocks without the key,
        # and decryption still "works".
        ct = modes.ecb_encrypt(SP800_PLAINTEXT, KEY_128)
        swapped = ct[16:32] + ct[0:16] + ct[32:]
        pt = modes.ecb_decrypt(swapped, KEY_128)
        assert pt == SP800_PLAINTEXT[16:32] + SP800_PLAINTEXT[0:16] + SP800_PLAINTEXT[32:]

    def test_round_trip_with_padding(self):
        rng = random.Random(3)
        for length in range(0, 50, 7):
            msg = random_bytes(rng, length)
            key = random_bytes(rng, 16)
            assert (
                modes.unpad(modes.ecb_decrypt(modes.ecb_encrypt(modes.pad(msg), key), key)) == msg
            )


# =============================================================================
# Step 4: Why ECB is broken
# =============================================================================

A, B, C = b"A" * 16, b"B" * 16, b"C" * 16


class TestRepeatedBlocks:
    @pytest.mark.parametrize(
        "blocks, count",
        [
            ([], 0),
            ([A], 0),
            ([A, B], 0),
            ([A, A], 1),
            ([A, B, A, A], 2),  # the example from the plan
            ([A, A, A, A], 3),
            ([A, B, C, A, B, C], 3),
            ([A, B, B, C, C, C], 3),
            ([A] * 10, 9),
        ],
    )
    def test_known(self, blocks, count):
        assert modes.repeated_blocks(b"".join(blocks)) == count

    def test_custom_block_size(self):
        assert modes.repeated_blocks(b"abcdabcdxyzw", 4) == 1  # abcd abcd xyzw
        assert modes.repeated_blocks(b"abcdabcdxyzw", 2) == 2  # ab cd ab cd xy zw
        assert modes.repeated_blocks(b"abcdabcdxyzw", 3) == 0  # abc dab cdx yzw
        assert modes.repeated_blocks(b"aaaaaa", 2) == 2

    def test_partial_overlap_is_not_a_repeat(self):
        # Only whole aligned blocks count: AAAA...B and AAAA...C differ in their last byte.
        assert modes.repeated_blocks(b"A" * 15 + b"B" + b"A" * 15 + b"C") == 0

    def test_accepts_bytearray(self):
        assert modes.repeated_blocks(bytearray(A + A)) == 1

    @pytest.mark.parametrize("length", [1, 15, 17, 33])
    def test_rejects_unaligned(self, length):
        with pytest.raises(ValueError):
            modes.repeated_blocks(bytes(length))

    def test_random_data_has_no_repeats(self):
        rng = random.Random(4)
        assert modes.repeated_blocks(random_bytes(rng, 16 * 64)) == 0


class TestECBLeaksPatterns:
    def test_ecb_keeps_plaintext_repeats(self):
        # Equal plaintext blocks -> equal ciphertext blocks, so the count survives encryption.
        plaintext = b"".join([A, B, A, A, C, B])
        assert modes.repeated_blocks(plaintext) == 3
        assert modes.repeated_blocks(modes.ecb_encrypt(plaintext, KEY_128)) == 3

    def test_repeats_are_in_the_same_positions(self):
        ct_blocks = modes.split_blocks(modes.ecb_encrypt(b"".join([A, B, A, A]), KEY_128))
        assert ct_blocks[0] == ct_blocks[2] == ct_blocks[3]
        assert ct_blocks[0] != ct_blocks[1]

    def test_leak_does_not_depend_on_the_key(self):
        rng = random.Random(5)
        plaintext = b"".join([A, B, A, C, A, B])
        for _ in range(5):
            key = random_bytes(rng, 16)
            assert modes.repeated_blocks(modes.ecb_encrypt(plaintext, key)) == 3

    def test_detect_ecb_without_the_key(self):
        # An attacker sees ciphertexts from an unknown key and spots the ECB one by its repeats.
        rng = random.Random(6)
        key = random_bytes(rng, 16)
        structured = A * 8 + random_bytes(rng, 64)  # e.g. a file with a run of identical blocks
        ecb_ct = modes.ecb_encrypt(structured, key)
        others = [random_bytes(rng, len(ecb_ct)) for _ in range(20)]
        candidates = others[:7] + [ecb_ct] + others[7:]
        guesses = [i for i, ct in enumerate(candidates) if modes.repeated_blocks(ct) > 0]
        assert guesses == [7]


# =============================================================================
# Step 5: CBC  (NIST SP 800-38A section 6.2, Appendix F.2)
# =============================================================================

SP800_IV = bytes(range(16))  # 000102...0f


class TestCBC:
    # F.2.1 / F.2.2: CBC-AES128
    CT_128 = bytes.fromhex(
        "7649abac8119b246cee98e9b12e9197d"
        "5086cb9b507219ee95db113a917678b2"
        "73bed6b8e3c1743b7116e69e22229516"
        "3ff1caa1681fac09120eca307586e1a7"
    )
    # F.2.5 / F.2.6: CBC-AES256
    CT_256 = bytes.fromhex(
        "f58c4c04d6e5f1ba779eabfb5f7bfbd6"
        "9cfc4e967edb808d679f777bc6702c7d"
        "39f23369a9d9bacfa530e26304231461"
        "b2eb05e2c39be9fcda6c19078c6a9d1b"
    )

    def test_encrypt_aes128(self):
        assert modes.cbc_encrypt(SP800_PLAINTEXT, KEY_128, SP800_IV) == self.CT_128

    def test_decrypt_aes128(self):
        assert modes.cbc_decrypt(self.CT_128, KEY_128, SP800_IV) == SP800_PLAINTEXT

    def test_encrypt_aes256(self):
        assert modes.cbc_encrypt(SP800_PLAINTEXT, KEY_256, SP800_IV) == self.CT_256

    def test_decrypt_aes256(self):
        assert modes.cbc_decrypt(self.CT_256, KEY_256, SP800_IV) == SP800_PLAINTEXT

    def test_first_block(self):
        # C_0 = AES(K, P_0 xor IV)
        from fulcrypt import aes

        expected = aes.encrypt_block(modes.xor_bytes(SP800_PLAINTEXT[:16], SP800_IV), KEY_128)
        assert modes.cbc_encrypt(SP800_PLAINTEXT[:16], KEY_128, SP800_IV) == expected

    def test_empty(self):
        assert modes.cbc_encrypt(b"", KEY_128, SP800_IV) == b""
        assert modes.cbc_decrypt(b"", KEY_128, SP800_IV) == b""

    def test_does_not_pad(self):
        assert len(modes.cbc_encrypt(bytes(32), KEY_128, SP800_IV)) == 32

    def test_returns_bytes(self):
        ct = modes.cbc_encrypt(bytearray(SP800_PLAINTEXT), KEY_128, bytearray(SP800_IV))
        assert type(ct) is bytes
        assert type(modes.cbc_decrypt(bytearray(ct), KEY_128, bytearray(SP800_IV))) is bytes

    @pytest.mark.parametrize("length", [1, 15, 17, 33])
    def test_rejects_unaligned(self, length):
        with pytest.raises(ValueError):
            modes.cbc_encrypt(bytes(length), KEY_128, SP800_IV)
        with pytest.raises(ValueError):
            modes.cbc_decrypt(bytes(length), KEY_128, SP800_IV)

    @pytest.mark.parametrize("iv_len", [0, 8, 15, 17, 32])
    def test_rejects_bad_iv(self, iv_len):
        with pytest.raises(ValueError):
            modes.cbc_encrypt(bytes(32), KEY_128, bytes(iv_len))
        with pytest.raises(ValueError):
            modes.cbc_decrypt(bytes(32), KEY_128, bytes(iv_len))

    def test_rejects_bad_key(self):
        with pytest.raises(ValueError):
            modes.cbc_encrypt(bytes(16), bytes(20), SP800_IV)

    def test_repeated_blocks_no_longer_repeat(self):
        # The ECB leak from Step 4 is gone:
        # chaining makes equal plaintext blocks encrypt differently.
        plaintext = b"A" * 16 * 8
        assert modes.repeated_blocks(modes.ecb_encrypt(plaintext, KEY_128)) == 7
        assert modes.repeated_blocks(modes.cbc_encrypt(plaintext, KEY_128, SP800_IV)) == 0

    def test_iv_changes_every_block(self):
        a = modes.split_blocks(modes.cbc_encrypt(SP800_PLAINTEXT, KEY_128, SP800_IV))
        b = modes.split_blocks(modes.cbc_encrypt(SP800_PLAINTEXT, KEY_128, bytes(16)))
        assert all(x != y for x, y in zip(a, b, strict=True))

    def test_same_iv_same_ciphertext(self):
        # Deterministic for a fixed IV -- which is exactly why IVs must not be reused (Step 6).
        assert modes.cbc_encrypt(SP800_PLAINTEXT, KEY_128, SP800_IV) == modes.cbc_encrypt(
            SP800_PLAINTEXT, KEY_128, SP800_IV
        )

    def test_prefix(self):
        # Block i depends only on blocks 0..i, so encrypting a prefix gives a prefix.
        full = modes.cbc_encrypt(SP800_PLAINTEXT, KEY_128, SP800_IV)
        assert modes.cbc_encrypt(SP800_PLAINTEXT[:32], KEY_128, SP800_IV) == full[:32]

    def test_decrypt_chains_with_previous_ciphertext(self):
        # P_i = AES^-1(C_i) xor C_{i-1}: any ciphertext block can serve as the IV for the rest.
        ct = self.CT_128
        assert modes.cbc_decrypt(ct[16:], KEY_128, ct[:16]) == SP800_PLAINTEXT[16:]
        assert modes.cbc_decrypt(ct[32:48], KEY_128, ct[16:32]) == SP800_PLAINTEXT[32:48]

    def test_wrong_iv_only_breaks_first_block(self):
        pt = modes.cbc_decrypt(self.CT_128, KEY_128, bytes(16))
        assert pt[:16] != SP800_PLAINTEXT[:16]
        assert pt[16:] == SP800_PLAINTEXT[16:]

    def test_round_trip_with_padding(self):
        rng = random.Random(7)
        for length in range(0, 50, 7):
            msg = random_bytes(rng, length)
            key, iv = random_bytes(rng, 16), random_bytes(rng, 16)
            ct = modes.cbc_encrypt(modes.pad(msg), key, iv)
            assert modes.unpad(modes.cbc_decrypt(ct, key, iv)) == msg


# =============================================================================
# Step 6: Using CBC safely
# =============================================================================
# The first three classes check your answers to the plan's questions 1-3 as properties of CBC.
# Try answering the questions before reading these tests: they contain the answers.


def flip_bit(data, bit):
    """Return data with one bit flipped (bit 0 = most significant bit of byte 0)."""
    out = bytearray(data)
    out[bit // 8] ^= 0x80 >> (bit % 8)
    return bytes(out)


def bit_diff(a, b):
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b, strict=True))


class TestIVReuse:
    """Question 1: same key and IV for two messages."""

    def test_equal_messages_give_equal_ciphertexts(self):
        # The attacker learns that two messages are identical.
        msg = SP800_PLAINTEXT
        assert modes.cbc_encrypt(msg, KEY_128, SP800_IV) == modes.cbc_encrypt(
            msg, KEY_128, SP800_IV
        )

    def test_common_prefix_is_visible(self):
        # Messages that agree on their first k blocks give ciphertexts that agree on k blocks,
        # and differ from the first differing block onwards.
        a = SP800_PLAINTEXT
        b = SP800_PLAINTEXT[:32] + bytes(32)
        ca = modes.split_blocks(modes.cbc_encrypt(a, KEY_128, SP800_IV))
        cb = modes.split_blocks(modes.cbc_encrypt(b, KEY_128, SP800_IV))
        assert ca[:2] == cb[:2]
        assert ca[2] != cb[2] and ca[3] != cb[3]

    def test_fresh_iv_hides_the_prefix(self):
        a = modes.cbc_encrypt(SP800_PLAINTEXT, KEY_128, SP800_IV)
        b = modes.cbc_encrypt(SP800_PLAINTEXT, KEY_128, bytes(16))
        assert all(
            x != y for x, y in zip(modes.split_blocks(a), modes.split_blocks(b), strict=True)
        )


class TestErrorPropagation:
    """Question 2: flip one bit in ciphertext block i."""

    CT = TestCBC.CT_128

    @pytest.mark.parametrize("block, bit", [(0, 0), (0, 77), (1, 5), (2, 127)])
    def test_garbles_block_i_flips_same_bit_in_block_i_plus_1(self, block, bit):
        bad = flip_bit(self.CT, 128 * block + bit)
        pt = modes.split_blocks(modes.cbc_decrypt(bad, KEY_128, SP800_IV))
        orig = modes.split_blocks(SP800_PLAINTEXT)
        for i in range(4):
            if i == block:
                assert bit_diff(pt[i], orig[i]) > 16, "block i decrypts to garbage"
            elif i == block + 1:
                assert pt[i] == flip_bit(orig[i], bit), "block i+1 has exactly the same bit flipped"
            else:
                assert pt[i] == orig[i], "all other blocks are unaffected"

    def test_last_block_only_garbles_itself(self):
        bad = flip_bit(self.CT, 128 * 3 + 9)
        pt = modes.cbc_decrypt(bad, KEY_128, SP800_IV)
        assert pt[:48] == SP800_PLAINTEXT[:48]
        assert pt[48:] != SP800_PLAINTEXT[48:]

    def test_flipping_the_iv_flips_the_first_block_only(self):
        pt = modes.cbc_decrypt(self.CT, KEY_128, flip_bit(SP800_IV, 3))
        assert pt == flip_bit(SP800_PLAINTEXT, 3)


class TestBitFlipping:
    """Question 3: change plaintext block i+1 to a chosen value without the key."""

    def test_attacker_sets_chosen_plaintext(self):
        key, iv = KEY_128, SP800_IV
        known = b"user=alice;role=" + b"guest;padding..."  # attacker knows block 1 is "guest;..."
        target = b"admin;padding..."
        target += b"." * (16 - len(target))
        ct = modes.cbc_encrypt(known, key, iv)

        # C_0' = C_0 xor P_1 xor P_1'  makes block 1 decrypt to P_1'.
        forged = modes.xor_bytes(ct[:16], modes.xor_bytes(known[16:32], target)) + ct[16:]
        pt = modes.cbc_decrypt(forged, key, iv)
        assert pt[16:] == target  # chosen change, exactly
        assert pt[:16] != known[:16]  # the price: block 0 is garbage

    def test_forged_iv_changes_first_block_cleanly(self):
        # With the IV sent alongside the ciphertext (Step 6 functions), the attacker can edit
        # block 0 too, and nothing else gets garbled.
        key = KEY_128
        msg = b"amount=00000100;"  # exactly one block
        data = modes.encrypt_message(msg, key)
        iv, ct = data[:16], data[16:]
        forged_iv = modes.xor_bytes(iv, modes.xor_bytes(msg, b"amount=99999999;"))
        assert modes.decrypt_message(forged_iv + ct, key) == b"amount=99999999;"


class TestMessageAPI:
    """encrypt_message / decrypt_message: random IV, padding, IV prepended."""

    @pytest.mark.parametrize("length", range(0, 65, 3))
    def test_round_trip(self, length):
        msg = bytes((5 * i) % 256 for i in range(length))
        assert modes.decrypt_message(modes.encrypt_message(msg, KEY_128), KEY_128) == msg

    @pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 32])
    def test_layout(self, length):
        # IV (16 bytes) || cbc_encrypt(pad(message), key, IV)
        msg = bytes(length)
        data = modes.encrypt_message(msg, KEY_128)
        assert len(data) == 16 + len(modes.pad(msg))
        iv, ct = data[:16], data[16:]
        assert modes.cbc_encrypt(modes.pad(msg), KEY_128, iv) == ct

    def test_returns_bytes(self):
        data = modes.encrypt_message(b"hello", KEY_128)
        assert type(data) is bytes
        assert type(modes.decrypt_message(data, KEY_128)) is bytes

    def test_same_message_encrypts_differently(self):
        msg = b"attack at dawn"
        outs = {modes.encrypt_message(msg, KEY_128) for _ in range(20)}
        assert len(outs) == 20
        assert len({o[:16] for o in outs}) == 20  # every IV is fresh

    def test_iv_comes_from_os_urandom(self, monkeypatch):
        calls = []

        def fake_urandom(n):
            calls.append(n)
            return bytes(range(0xA0, 0xA0 + n))

        monkeypatch.setattr(os, "urandom", fake_urandom)
        if hasattr(modes, "urandom"):  # also covers `from os import urandom`
            monkeypatch.setattr(modes, "urandom", fake_urandom)
        data = modes.encrypt_message(b"hello", KEY_128)
        assert calls == [16]
        assert data[:16] == bytes(range(0xA0, 0xB0))

    def test_aes256_key(self):
        msg = b"a message for a 256-bit key"
        assert modes.decrypt_message(modes.encrypt_message(msg, KEY_256), KEY_256) == msg

    @pytest.mark.parametrize("length", [0, 15, 16, 31, 33])
    def test_decrypt_rejects_bad_length(self, length):
        # Valid data is IV + at least one block, and block-aligned.
        with pytest.raises(ValueError):
            modes.decrypt_message(bytes(length), KEY_128)

    def test_decrypt_rejects_tampered_padding(self):
        data = modes.encrypt_message(b"hello", KEY_128)
        # Flipping the last byte of the second-to-last block (here: the IV)
        # changes the padding byte.
        bad = flip_bit(data, 8 * 15 + 7)
        with pytest.raises(ValueError):
            modes.decrypt_message(bad, KEY_128)

    def test_wrong_key_does_not_return_the_message(self):
        data = modes.encrypt_message(b"secret message", KEY_128)
        try:
            out = modes.decrypt_message(data, bytes(16))
        except ValueError:
            return  # most likely: invalid padding
        assert out != b"secret message"


# =============================================================================
# Part 1, Step 1: CTR  (NIST SP 800-38A section 6.5, Appendix F.5)
# =============================================================================

SP800_COUNTER = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")


class TestIncCounter:
    @pytest.mark.parametrize(
        "block, bits, expected",
        [
            ("00" * 16, 128, "00" * 15 + "01"),
            ("00" * 15 + "ff", 128, "00" * 14 + "0100"),  # carry within the counter
            ("ff" * 16, 128, "00" * 16),  # wraps mod 2^128
            ("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff", 128, "f0f1f2f3f4f5f6f7f8f9fafbfcfdff00"),
            ("aa" * 12 + "00000001", 32, "aa" * 12 + "00000002"),
            ("aa" * 12 + "ffffffff", 32, "aa" * 12 + "00000000"),  # wraps, IV part untouched
            ("aa" * 8 + "ffffffffffffffff", 64, "aa" * 8 + "0000000000000000"),
            ("aa" * 15 + "ff", 8, "aa" * 15 + "00"),
            # Counters that don't end on a byte boundary: the IV bits sharing the top
            # counter byte must survive.
            ("aa" * 15 + "a5", 4, "aa" * 15 + "a6"),
            ("aa" * 15 + "af", 4, "aa" * 15 + "a0"),  # 4-bit wrap, high nibble kept
            ("aa" * 14 + "afff", 12, "aa" * 14 + "a000"),
            ("ff" * 16, 100, "fffffff" + "0" * 25),  # top 28 bits are IV
        ],
    )
    def test_known(self, block, bits, expected):
        assert modes.inc_counter(bytes.fromhex(block), bits).hex() == expected

    def test_default_is_128_bits(self):
        assert modes.inc_counter(b"\xff" * 16) == bytes(16)

    def test_returns_bytes(self):
        assert type(modes.inc_counter(bytearray(16))) is bytes

    @pytest.mark.parametrize("bits", [0, -8, 129])
    def test_rejects_bad_bits(self, bits):
        with pytest.raises(ValueError):
            modes.inc_counter(bytes(16), bits)

    @pytest.mark.parametrize("length", [0, 15, 17])
    def test_rejects_bad_block(self, length):
        with pytest.raises(ValueError):
            modes.inc_counter(bytes(length))


class TestCTR:
    # F.5.1 / F.5.2: CTR-AES128
    CT_128 = bytes.fromhex(
        "874d6191b620e3261bef6864990db6ce"
        "9806f66b7970fdff8617187bb9fffdff"
        "5ae4df3edbd5d35e5b4f09020db03eab"
        "1e031dda2fbe03d1792170a0f3009cee"
    )
    # F.5.5 / F.5.6: CTR-AES256
    CT_256 = bytes.fromhex(
        "601ec313775789a5b7a7f504bbf3d228"
        "f443e3ca4d62b59aca84e990cacaf5c5"
        "2b0930daa23de94ce87017ba2d84988d"
        "dfc9c58db67aada613c2dd08457941a6"
    )

    def test_encrypt_aes128(self):
        assert modes.ctr_encrypt(SP800_PLAINTEXT, KEY_128, SP800_COUNTER) == self.CT_128

    def test_decrypt_aes128(self):
        assert modes.ctr_decrypt(self.CT_128, KEY_128, SP800_COUNTER) == SP800_PLAINTEXT

    def test_encrypt_aes256(self):
        assert modes.ctr_encrypt(SP800_PLAINTEXT, KEY_256, SP800_COUNTER) == self.CT_256

    def test_decrypt_aes256(self):
        assert modes.ctr_decrypt(self.CT_256, KEY_256, SP800_COUNTER) == SP800_PLAINTEXT

    def test_first_keystream_block_is_aes_of_counter(self):
        from fulcrypt import aes

        keystream = modes.ctr_encrypt(bytes(16), KEY_128, SP800_COUNTER)
        assert keystream == aes.encrypt_block(SP800_COUNTER, KEY_128)

    @pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 31, 33, 50, 64])
    def test_any_length_no_padding(self, length):
        # A stream cipher: output length == input length, a partial last block is fine.
        data = SP800_PLAINTEXT[:length]
        ct = modes.ctr_encrypt(data, KEY_128, SP800_COUNTER)
        assert len(ct) == length
        assert ct == self.CT_128[:length]  # a prefix of the full ciphertext
        assert modes.ctr_decrypt(ct, KEY_128, SP800_COUNTER) == data

    def test_decrypt_is_encrypt(self):
        rng = random.Random(20)
        data = random_bytes(rng, 45)
        assert modes.ctr_decrypt(data, KEY_128, SP800_COUNTER) == modes.ctr_encrypt(
            data, KEY_128, SP800_COUNTER
        )

    def test_counter_bits_32_wraps_without_touching_iv(self):
        # GCM-style counter: IV (96 bits) || counter (32 bits). The second keystream block must
        # come from IV || 00000000, not from a carry into the IV. Expected value from a reference.
        iv = bytes.fromhex("cafebabefacedbaddecaf888")
        ks = modes.ctr_encrypt(bytes(32), KEY_128, iv + b"\xff\xff\xff\xff", counter_bits=32)
        assert ks[16:].hex() == "8d9e5b8d69f60625ec32f971a71ce8b0"

    def test_counter_bits_change_the_result(self):
        start = b"\x00" * 12 + b"\xff\xff\xff\xff"
        a = modes.ctr_encrypt(bytes(32), KEY_128, start, counter_bits=32)
        b = modes.ctr_encrypt(bytes(32), KEY_128, start, counter_bits=128)
        assert a[:16] == b[:16]  # same first counter block
        assert a[16:] != b[16:]  # 32-bit wraps to 0, 128-bit carries into byte 11

    def test_random_access(self):
        # Block i can be decrypted alone by starting the counter at T_0 + i.
        t2 = modes.inc_counter(modes.inc_counter(SP800_COUNTER))
        assert modes.ctr_decrypt(self.CT_128[32:48], KEY_128, t2) == SP800_PLAINTEXT[32:48]

    def test_returns_bytes(self):
        out = modes.ctr_encrypt(bytearray(b"abc"), KEY_128, bytearray(SP800_COUNTER))
        assert type(out) is bytes

    @pytest.mark.parametrize("length", [0, 12, 15, 17])
    def test_rejects_bad_counter_block(self, length):
        with pytest.raises(ValueError):
            modes.ctr_encrypt(b"abc", KEY_128, bytes(length))

    def test_rejects_bad_key(self):
        with pytest.raises(ValueError):
            modes.ctr_encrypt(b"abc", bytes(20), SP800_COUNTER)


class TestCTRProperties:
    """Why the counter block must never repeat under one key, and why CTR alone is not enough."""

    def test_nonce_reuse_leaks_xor_of_plaintexts(self):
        # Same key and counter -> same keystream: C1 xor C2 = P1 xor P2 (a "two-time pad").
        p1 = b"attack at dawn!! meet at the old bridge"
        p2 = b"retreat at noon. bring the documents..."
        c1 = modes.ctr_encrypt(p1, KEY_128, SP800_COUNTER)
        c2 = modes.ctr_encrypt(p2, KEY_128, SP800_COUNTER)
        assert modes.xor_bytes(c1, c2) == modes.xor_bytes(p1, p2)

    def test_known_plaintext_reveals_the_other_message(self):
        p1 = b"known plaintext, e.g. a fixed header.."
        p2 = b"the secret message under the same nonce"[: len(p1)]
        c1 = modes.ctr_encrypt(p1, KEY_128, SP800_COUNTER)
        c2 = modes.ctr_encrypt(p2, KEY_128, SP800_COUNTER)
        keystream = modes.xor_bytes(c1, p1)
        assert modes.xor_bytes(c2, keystream) == p2

    @pytest.mark.parametrize("bit", [0, 7, 100, 300])
    def test_bit_flip_flips_exactly_that_bit(self, bit):
        # Fully malleable, and unlike CBC nothing gets garbled: CTR needs a MAC.
        ct = modes.ctr_encrypt(SP800_PLAINTEXT, KEY_128, SP800_COUNTER)
        pt = modes.ctr_decrypt(flip_bit(ct, bit), KEY_128, SP800_COUNTER)
        assert pt == flip_bit(SP800_PLAINTEXT, bit)


# =============================================================================
# Part 1, Step 2: OFB  (NIST SP 800-38A section 6.4, Appendix F.4)
# =============================================================================


class TestOFB:
    # F.4.1 / F.4.2: OFB-AES128, IV = 000102...0f
    CT_128 = bytes.fromhex(
        "3b3fd92eb72dad20333449f8e83cfb4a"
        "7789508d16918f03f53c52dac54ed825"
        "9740051e9c5fecf64344f7a82260edcc"
        "304c6528f659c77866a510d9c1d6ae5e"
    )
    # F.4.5 / F.4.6: OFB-AES256
    CT_256 = bytes.fromhex(
        "dc7e84bfda79164b7ecd8486985d3860"
        "4febdc6740d20b3ac88f6ad82a4fb08d"
        "71ab47a086e86eedf39d1c5bba97c408"
        "0126141d67f37be8538f5a8be740e484"
    )

    def test_encrypt_aes128(self):
        assert modes.ofb_encrypt(SP800_PLAINTEXT, KEY_128, SP800_IV) == self.CT_128

    def test_decrypt_aes128(self):
        assert modes.ofb_decrypt(self.CT_128, KEY_128, SP800_IV) == SP800_PLAINTEXT

    def test_encrypt_aes256(self):
        assert modes.ofb_encrypt(SP800_PLAINTEXT, KEY_256, SP800_IV) == self.CT_256

    def test_decrypt_aes256(self):
        assert modes.ofb_decrypt(self.CT_256, KEY_256, SP800_IV) == SP800_PLAINTEXT

    def test_keystream_chains_through_aes(self):
        # Encrypting zeros reveals the keystream: O_1 = AES(K, IV), O_{i+1} = AES(K, O_i).
        from fulcrypt import aes

        ks = modes.split_blocks(modes.ofb_encrypt(bytes(48), KEY_128, SP800_IV))
        assert ks[0] == aes.encrypt_block(SP800_IV, KEY_128)
        assert ks[1] == aes.encrypt_block(ks[0], KEY_128)
        assert ks[2] == aes.encrypt_block(ks[1], KEY_128)

    def test_differs_from_ctr(self):
        # Same first keystream block as CTR started at the IV, different after that.
        ofb = modes.ofb_encrypt(bytes(32), KEY_128, SP800_IV)
        ctr = modes.ctr_encrypt(bytes(32), KEY_128, SP800_IV)
        assert ofb[:16] == ctr[:16]
        assert ofb[16:] != ctr[16:]

    @pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 31, 33, 50, 64])
    def test_any_length_no_padding(self, length):
        data = SP800_PLAINTEXT[:length]
        ct = modes.ofb_encrypt(data, KEY_128, SP800_IV)
        assert ct == self.CT_128[:length]
        assert modes.ofb_decrypt(ct, KEY_128, SP800_IV) == data

    def test_decrypt_is_encrypt(self):
        rng = random.Random(21)
        data = random_bytes(rng, 45)
        assert modes.ofb_decrypt(data, KEY_128, SP800_IV) == modes.ofb_encrypt(
            data, KEY_128, SP800_IV
        )

    def test_returns_bytes(self):
        out = modes.ofb_encrypt(bytearray(b"abc"), KEY_128, bytearray(SP800_IV))
        assert type(out) is bytes

    @pytest.mark.parametrize("iv_len", [0, 8, 15, 17])
    def test_rejects_bad_iv(self, iv_len):
        for data in (b"", b"abc"):  # also with empty data, where the loop never runs
            with pytest.raises(ValueError):
                modes.ofb_encrypt(data, KEY_128, bytes(iv_len))
            with pytest.raises(ValueError):
                modes.ofb_decrypt(data, KEY_128, bytes(iv_len))

    def test_rejects_bad_key(self):
        with pytest.raises(ValueError):
            modes.ofb_encrypt(b"abc", bytes(20), SP800_IV)


class TestOFBProperties:
    """The keystream depends only on key and IV, never on the data."""

    def test_keystream_independent_of_plaintext(self):
        # C = P xor O for any P: the same O for every message under one (key, IV).
        rng = random.Random(22)
        keystream = modes.ofb_encrypt(bytes(40), KEY_128, SP800_IV)
        for _ in range(5):
            p = random_bytes(rng, 40)
            assert modes.ofb_encrypt(p, KEY_128, SP800_IV) == modes.xor_bytes(p, keystream)

    def test_iv_reuse_leaks_xor_of_plaintexts(self):
        p1 = b"OFB with a reused IV is a two-time pad."
        p2 = b"so is CTR with a reused counter block!!"
        c1 = modes.ofb_encrypt(p1, KEY_128, SP800_IV)
        c2 = modes.ofb_encrypt(p2, KEY_128, SP800_IV)
        assert modes.xor_bytes(c1, c2) == modes.xor_bytes(p1, p2)

    @pytest.mark.parametrize("bit", [0, 9, 200, 511])
    def test_bit_flip_flips_exactly_that_bit(self, bit):
        ct = modes.ofb_encrypt(SP800_PLAINTEXT, KEY_128, SP800_IV)
        pt = modes.ofb_decrypt(flip_bit(ct, bit), KEY_128, SP800_IV)
        assert pt == flip_bit(SP800_PLAINTEXT, bit)
