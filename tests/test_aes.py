"""Tests for AES (FIPS-197). Built step by step; see PLANNING.md.

Design: bytes inside the cipher are GF(2^8) field elements, the state is state[r][c]
(4 rows x 4 columns, filled column by column), and every transformation returns a new state.
"""

import os
import random

import pytest

import aes
from aes import bytes_to_state, state_to_bytes
from elements import GF2n

AES = GF2n(0x11B)

# FIPS-197 Appendix B input block
FIPS_INPUT = bytes.fromhex("3243f6a8885a308d313198a2e0370734")


def random_block(rng):
    return bytes(rng.randrange(256) for _ in range(16))


# =============================================================================
# Step 1: bytes <-> state
# =============================================================================

class TestBytesToState:
    def test_shape(self):
        state = bytes_to_state(FIPS_INPUT)
        assert len(state) == 4
        assert all(len(row) == 4 for row in state)

    def test_entries_are_field_elements(self):
        state = bytes_to_state(FIPS_INPUT)
        assert all(type(x) is AES for row in state for x in row)

    def test_column_major(self):
        # in[r + 4c] -> s[r][c]  (FIPS-197 section 3.4)
        block = bytes(range(16))
        state = bytes_to_state(block)
        for i in range(16):
            assert state[i % 4][i // 4] == AES(i), f"byte {i} should be at [{i % 4}][{i // 4}]"

    def test_first_column_is_first_word(self):
        state = bytes_to_state(FIPS_INPUT)
        assert [state[r][0] for r in range(4)] == [AES(0x32), AES(0x43), AES(0xF6), AES(0xA8)]

    def test_first_row(self):
        # Row 0 takes bytes 0, 4, 8, 12.
        state = bytes_to_state(FIPS_INPUT)
        assert list(state[0]) == [AES(0x32), AES(0x88), AES(0x31), AES(0xE0)]

    def test_rows_are_separate_objects(self):
        # Catches the [[...] * 4] * 4 trap: four references to one row.
        state = bytes_to_state(FIPS_INPUT)
        assert len({id(row) for row in state}) == 4

    def test_accepts_bytearray(self):
        assert bytes_to_state(bytearray(FIPS_INPUT)) == bytes_to_state(FIPS_INPUT)

    @pytest.mark.parametrize("length", [0, 1, 15, 17, 32])
    def test_rejects_wrong_length(self, length):
        with pytest.raises(ValueError):
            bytes_to_state(bytes(length))


class TestStateToBytes:
    def test_returns_bytes(self):
        assert type(state_to_bytes(bytes_to_state(FIPS_INPUT))) is bytes

    def test_known(self):
        state = [[AES(r + 4 * c) for c in range(4)] for r in range(4)]
        assert state_to_bytes(state) == bytes(range(16))

    def test_round_trip_fips(self):
        assert state_to_bytes(bytes_to_state(FIPS_INPUT)) == FIPS_INPUT

    def test_round_trip_random(self):
        rng = random.Random(1)
        for _ in range(200):
            block = random_block(rng)
            assert state_to_bytes(bytes_to_state(block)) == block

    def test_round_trip_edges(self):
        for block in (bytes(16), b"\xff" * 16, os.urandom(16)):
            assert state_to_bytes(bytes_to_state(block)) == block

    def test_does_not_modify_state(self):
        state = bytes_to_state(FIPS_INPUT)
        before = [list(row) for row in state]
        state_to_bytes(state)
        assert [list(row) for row in state] == before


# =============================================================================
# Step 2: SubBytes  (FIPS-197 section 5.1.1, 5.3.2)
# =============================================================================

def state_from_hex(h):
    return bytes_to_state(bytes.fromhex(h))


def hex_of(state):
    return state_to_bytes(state).hex()


class TestSboxTables:
    def test_sizes(self):
        assert len(aes.sbox) == 256
        assert len(aes.inv_sbox) == 256

    def test_entries_are_bytes(self):
        assert all(type(v) is int and 0 <= v < 256 for v in aes.sbox)
        assert all(type(v) is int and 0 <= v < 256 for v in aes.inv_sbox)

    def test_first_row(self):
        assert bytes(aes.sbox[:16]).hex() == "637c777bf26b6fc53001672bfed7ab76"

    def test_inverse_first_row(self):
        assert bytes(aes.inv_sbox[:16]).hex() == "52096ad53036a538bf40a39e81f3d7fb"

    @pytest.mark.parametrize("x, y", [
        (0x00, 0x63),     # 0 has no inverse; only the affine step acts
        (0x01, 0x7C),
        (0x53, 0xED),     # FIPS-197 example in section 5.1.1
        (0x10, 0xCA),
        (0xC9, 0xDD),
        (0xFF, 0x16),
    ])
    def test_known(self, x, y):
        assert aes.sbox[x] == y
        assert aes.inv_sbox[y] == x

    def test_is_permutation(self):
        assert sorted(aes.sbox) == list(range(256))
        assert sorted(aes.inv_sbox) == list(range(256))

    def test_inverse_tables(self):
        for x in range(256):
            assert aes.inv_sbox[aes.sbox[x]] == x
            assert aes.sbox[aes.inv_sbox[x]] == x

    def test_no_fixed_points(self):
        # A design goal of the affine step: S(x) != x and S(x) != ~x for every byte.
        for x in range(256):
            assert aes.sbox[x] != x
            assert aes.sbox[x] != x ^ 0xFF

    def test_not_affine(self):
        # The field inverse makes the S-box nonlinear: S(a ^ b) != S(a) ^ S(b) ^ S(0) in general.
        assert any(
            aes.sbox[a ^ b] != aes.sbox[a] ^ aes.sbox[b] ^ aes.sbox[0]
            for a in range(1, 16) for b in range(1, 16)
        )


class TestSubBytes:
    # FIPS-197 Appendix B, round 1: "Start of Round" -> "After SubBytes"
    ROUND1_START = "193de3bea0f4e22b9ac68d2ae9f84808"
    ROUND1_AFTER = "d42711aee0bf98f1b8b45de51e415230"

    def test_appendix_b_round1(self):
        assert hex_of(aes.sub_bytes(state_from_hex(self.ROUND1_START))) == self.ROUND1_AFTER

    def test_inverse_appendix_b_round1(self):
        assert hex_of(aes.inv_sub_bytes(state_from_hex(self.ROUND1_AFTER))) == self.ROUND1_START

    def test_entries_stay_field_elements(self):
        out = aes.sub_bytes(state_from_hex(self.ROUND1_START))
        assert all(type(x) is AES for row in out for x in row)
        out = aes.inv_sub_bytes(state_from_hex(self.ROUND1_START))
        assert all(type(x) is AES for row in out for x in row)

    def test_returns_new_state(self):
        state = state_from_hex(self.ROUND1_START)
        before = [list(row) for row in state]
        out = aes.sub_bytes(state)
        assert out is not state
        assert [list(row) for row in state] == before
        aes.inv_sub_bytes(state)
        assert [list(row) for row in state] == before

    def test_bytewise(self):
        # Each byte is substituted on its own: position doesn't matter.
        block = bytes(range(16))
        assert state_to_bytes(aes.sub_bytes(bytes_to_state(block))) == bytes(aes.sbox[b] for b in block)

    def test_round_trip_random(self):
        rng = random.Random(2)
        for _ in range(100):
            state = bytes_to_state(random_block(rng))
            assert aes.inv_sub_bytes(aes.sub_bytes(state)) == state
            assert aes.sub_bytes(aes.inv_sub_bytes(state)) == state


# =============================================================================
# Step 3: ShiftRows  (FIPS-197 section 5.1.2, 5.3.1)
# =============================================================================

def labelled_state():
    # Entry [r][c] holds the byte 0xRC, so positions are easy to read in failures.
    return [[AES(16 * r + c) for c in range(4)] for r in range(4)]


class TestShiftRows:
    # FIPS-197 Appendix B, round 1: "After SubBytes" -> "After ShiftRows"
    ROUND1_BEFORE = "d42711aee0bf98f1b8b45de51e415230"
    ROUND1_AFTER = "d4bf5d30e0b452aeb84111f11e2798e5"

    def test_appendix_b_round1(self):
        assert hex_of(aes.shift_rows(state_from_hex(self.ROUND1_BEFORE))) == self.ROUND1_AFTER

    def test_inverse_appendix_b_round1(self):
        assert hex_of(aes.inv_shift_rows(state_from_hex(self.ROUND1_AFTER))) == self.ROUND1_BEFORE

    def test_rotates_each_row_left_by_its_index(self):
        # s'[r][c] = s[r][(c + r) mod 4]
        s = labelled_state()
        out = aes.shift_rows(s)
        for r in range(4):
            for c in range(4):
                assert out[r][c] == s[r][(c + r) % 4], f"[{r}][{c}]"

    def test_inverse_rotates_right(self):
        # s'[r][c] = s[r][(c - r) mod 4]
        s = labelled_state()
        out = aes.inv_shift_rows(s)
        for r in range(4):
            for c in range(4):
                assert out[r][c] == s[r][(c - r) % 4], f"[{r}][{c}]"

    def test_row_zero_unchanged(self):
        s = labelled_state()
        assert list(aes.shift_rows(s)[0]) == list(s[0])
        assert list(aes.inv_shift_rows(s)[0]) == list(s[0])

    def test_bytes_stay_in_their_row(self):
        s = labelled_state()
        out = aes.shift_rows(s)
        for r in range(4):
            assert sorted(x.bits for x in out[r]) == sorted(x.bits for x in s[r])

    def test_spreads_columns(self):
        # After ShiftRows every column holds one byte from each original column.
        # This is what lets MixColumns mix bytes from different columns in the next round.
        s = labelled_state()
        out = aes.shift_rows(s)
        for c in range(4):
            original_columns = {out[r][c].bits % 16 for r in range(4)}
            assert original_columns == {0, 1, 2, 3}

    def test_four_times_is_identity(self):
        s = labelled_state()
        out = s
        for _ in range(4):
            out = aes.shift_rows(out)
        assert out == s

    def test_entries_stay_field_elements(self):
        for f in (aes.shift_rows, aes.inv_shift_rows):
            assert all(type(x) is AES for row in f(labelled_state()) for x in row)

    def test_returns_new_state(self):
        s = labelled_state()
        before = [list(row) for row in s]
        for f in (aes.shift_rows, aes.inv_shift_rows):
            out = f(s)
            assert out is not s
            assert all(out[r] is not s[r] for r in range(4))
            assert [list(row) for row in s] == before

    def test_round_trip_random(self):
        rng = random.Random(3)
        for _ in range(100):
            s = bytes_to_state(random_block(rng))
            assert aes.inv_shift_rows(aes.shift_rows(s)) == s
            assert aes.shift_rows(aes.inv_shift_rows(s)) == s


# =============================================================================
# Step 4: MixColumns  (FIPS-197 section 5.1.3, 5.3.3)
# =============================================================================

def state_with_column(c, column_hex, fill=0):
    """A state whose column c is the given 4 bytes; all other entries are `fill`."""
    s = [[AES(fill) for _ in range(4)] for _ in range(4)]
    for r, b in enumerate(bytes.fromhex(column_hex)):
        s[r][c] = AES(b)
    return s


def column_hex(state, c):
    return bytes(state[r][c].bits for r in range(4)).hex()


class TestMixColumns:
    # FIPS-197 Appendix B, round 1: "After ShiftRows" -> "After MixColumns"
    ROUND1_BEFORE = "d4bf5d30e0b452aeb84111f11e2798e5"
    ROUND1_AFTER = "046681e5e0cb199a48f8d37a2806264c"

    def test_appendix_b_round1(self):
        assert hex_of(aes.mix_columns(state_from_hex(self.ROUND1_BEFORE))) == self.ROUND1_AFTER

    def test_inverse_appendix_b_round1(self):
        assert hex_of(aes.inv_mix_columns(state_from_hex(self.ROUND1_AFTER))) == self.ROUND1_BEFORE

    @pytest.mark.parametrize("before, after", [
        ("db135345", "8e4da1bc"),
        ("f20a225c", "9fdc589d"),
        ("01010101", "01010101"),
        ("c6c6c6c6", "c6c6c6c6"),   # equal bytes: 2 + 3 + 1 + 1 = 1 in GF(2^8)
        ("d4d4d4d5", "d5d5d7d6"),
        ("2d26314c", "4d7ebdf8"),
    ])
    @pytest.mark.parametrize("c", range(4))
    def test_single_column(self, before, after, c):
        # Same column vector in every position c gives the same result there.
        out = aes.mix_columns(state_with_column(c, before))
        assert column_hex(out, c) == after
        back = aes.inv_mix_columns(state_with_column(c, after))
        assert column_hex(back, c) == before

    def test_columns_are_independent(self):
        # Changing column 0 must not affect columns 1-3: the matrix acts on columns, not rows.
        rng = random.Random(4)
        s = bytes_to_state(random_block(rng))
        t = [list(row) for row in s]
        for r in range(4):
            t[r][0] = AES(t[r][0].bits ^ 0xFF)
        out_s, out_t = aes.mix_columns(s), aes.mix_columns(t)
        for c in (1, 2, 3):
            assert column_hex(out_s, c) == column_hex(out_t, c)
        assert column_hex(out_s, 0) != column_hex(out_t, 0)

    def test_one_byte_changes_whole_column(self):
        # Diffusion: changing a single input byte changes all 4 output bytes of its column.
        rng = random.Random(5)
        for _ in range(50):
            s = bytes_to_state(random_block(rng))
            r, c = rng.randrange(4), rng.randrange(4)
            t = [list(row) for row in s]
            t[r][c] = AES(t[r][c].bits ^ rng.randrange(1, 256))
            out_s, out_t = aes.mix_columns(s), aes.mix_columns(t)
            assert all(out_s[i][c] != out_t[i][c] for i in range(4))

    def test_linear(self):
        # MixColumns is linear over GF(2^8): mix(a + b) = mix(a) + mix(b).
        rng = random.Random(6)
        for _ in range(50):
            a = bytes_to_state(random_block(rng))
            b = bytes_to_state(random_block(rng))
            a_plus_b = [[a[r][c] + b[r][c] for c in range(4)] for r in range(4)]
            ma, mb = aes.mix_columns(a), aes.mix_columns(b)
            assert aes.mix_columns(a_plus_b) == [[ma[r][c] + mb[r][c] for c in range(4)] for r in range(4)]

    def test_four_times_is_identity(self):
        # The AES matrix M satisfies M^4 = I, so the inverse is M^3.
        rng = random.Random(7)
        s = bytes_to_state(random_block(rng))
        out = s
        for _ in range(4):
            out = aes.mix_columns(out)
        assert out == s
        assert aes.inv_mix_columns(s) == aes.mix_columns(aes.mix_columns(aes.mix_columns(s)))

    def test_entries_stay_field_elements(self):
        for f in (aes.mix_columns, aes.inv_mix_columns):
            assert all(type(x) is AES for row in f(state_from_hex(self.ROUND1_BEFORE)) for x in row)

    def test_returns_new_state(self):
        s = state_from_hex(self.ROUND1_BEFORE)
        before = [list(row) for row in s]
        for f in (aes.mix_columns, aes.inv_mix_columns):
            out = f(s)
            assert out is not s
            assert all(out[r] is not s[r] for r in range(4))
            assert [list(row) for row in s] == before

    def test_round_trip_random(self):
        rng = random.Random(8)
        for _ in range(100):
            s = bytes_to_state(random_block(rng))
            assert aes.inv_mix_columns(aes.mix_columns(s)) == s
            assert aes.mix_columns(aes.inv_mix_columns(s)) == s


# =============================================================================
# Step 5: AddRoundKey  (FIPS-197 section 5.1.4)
# =============================================================================

def words_from_hex(h):
    """16 bytes -> 4 words of 4 field elements (word c = bytes 4c .. 4c+3)."""
    b = bytes.fromhex(h)
    return [[AES(b[4 * c + r]) for r in range(4)] for c in range(4)]


class TestAddRoundKey:
    # FIPS-197 Appendix B
    INPUT = "3243f6a8885a308d313198a2e0370734"
    KEY = "2b7e151628aed2a6abf7158809cf4f3c"          # round key 0 = the cipher key
    ROUND1_START = "193de3bea0f4e22b9ac68d2ae9f84808"
    ROUND1_MIXED = "046681e5e0cb199a48f8d37a2806264c"  # after MixColumns in round 1
    ROUND_KEY_1 = "a0fafe1788542cb123a339392a6c7605"
    ROUND2_START = "a49c7ff2689f352b6b5bea43026a5049"

    def test_initial_round_key(self):
        out = aes.add_round_key(state_from_hex(self.INPUT), words_from_hex(self.KEY))
        assert hex_of(out) == self.ROUND1_START

    def test_round_1_key(self):
        out = aes.add_round_key(state_from_hex(self.ROUND1_MIXED), words_from_hex(self.ROUND_KEY_1))
        assert hex_of(out) == self.ROUND2_START

    def test_word_c_goes_to_column_c(self):
        # Only word 2 is nonzero, so only column 2 may change.
        s = labelled_state()
        key = [[AES(0) for _ in range(4)] for _ in range(4)]
        key[2] = [AES(0x10), AES(0x20), AES(0x30), AES(0x40)]
        out = aes.add_round_key(s, key)
        for r in range(4):
            for c in range(4):
                expected = s[r][c] + key[2][r] if c == 2 else s[r][c]
                assert out[r][c] == expected, f"[{r}][{c}]"

    def test_word_index_is_row(self):
        # round_key[c][r] is added to state[r][c] -- a word is a column, indexed by row.
        s = bytes_to_state(bytes(16))
        key = [[AES(16 * c + r) for r in range(4)] for c in range(4)]
        out = aes.add_round_key(s, key)
        for r in range(4):
            for c in range(4):
                assert out[r][c] == AES(16 * c + r), f"[{r}][{c}]"

    def test_zero_key_is_identity(self):
        s = state_from_hex(self.INPUT)
        assert aes.add_round_key(s, words_from_hex("00" * 16)) == s

    def test_is_own_inverse(self):
        rng = random.Random(9)
        for _ in range(100):
            s = bytes_to_state(random_block(rng))
            key = words_from_hex(random_block(rng).hex())
            assert aes.add_round_key(aes.add_round_key(s, key), key) == s

    def test_same_as_bytewise_xor(self):
        rng = random.Random(10)
        for _ in range(50):
            block, key = random_block(rng), random_block(rng)
            out = aes.add_round_key(bytes_to_state(block), words_from_hex(key.hex()))
            assert state_to_bytes(out) == bytes(a ^ b for a, b in zip(block, key))

    def test_entries_stay_field_elements(self):
        out = aes.add_round_key(state_from_hex(self.INPUT), words_from_hex(self.KEY))
        assert all(type(x) is AES for row in out for x in row)

    def test_returns_new_state(self):
        s = state_from_hex(self.INPUT)
        key = words_from_hex(self.KEY)
        s_before = [list(row) for row in s]
        key_before = [list(w) for w in key]
        out = aes.add_round_key(s, key)
        assert out is not s
        assert all(out[r] is not s[r] for r in range(4))
        assert [list(row) for row in s] == s_before
        assert [list(w) for w in key] == key_before


# =============================================================================
# Step 6: Key expansion  (FIPS-197 section 5.2, Appendix A.1)
# =============================================================================

def word(h):
    return [AES(b) for b in bytes.fromhex(h)]


def word_hex(w):
    return bytes(x.bits for x in w).hex()


class TestRotWord:
    def test_known(self):
        # FIPS-197 A.1, i = 4
        assert word_hex(aes.rot_word(word("09cf4f3c"))) == "cf4f3c09"

    def test_rotates_left_by_one(self):
        a, b, c, d = word("01020304")
        assert aes.rot_word([a, b, c, d]) == [b, c, d, a]

    def test_four_times_is_identity(self):
        w = word("a0fafe17")
        out = w
        for _ in range(4):
            out = aes.rot_word(out)
        assert out == w

    def test_returns_new_list(self):
        w = word("09cf4f3c")
        before = list(w)
        out = aes.rot_word(w)
        assert out is not w
        assert w == before


class TestSubWord:
    def test_known(self):
        # FIPS-197 A.1, i = 4: after RotWord -> after SubWord
        assert word_hex(aes.sub_word(word("cf4f3c09"))) == "8a84eb01"

    def test_uses_sbox(self):
        w = word("00015310")
        assert aes.sub_word(w) == [AES(aes.sbox[x.bits]) for x in w]

    def test_entries_stay_field_elements(self):
        assert all(type(x) is AES for x in aes.sub_word(word("cf4f3c09")))

    def test_returns_new_list(self):
        w = word("cf4f3c09")
        before = list(w)
        out = aes.sub_word(w)
        assert out is not w
        assert w == before


class TestRcon:
    @pytest.mark.parametrize("i, first", [
        (1, 0x01), (2, 0x02), (3, 0x04), (4, 0x08), (5, 0x10),
        (6, 0x20), (7, 0x40), (8, 0x80),
        (9, 0x1B),    # x^8 reduced mod x^8 + x^4 + x^3 + x + 1
        (10, 0x36),
    ])
    def test_known(self, i, first):
        assert aes.rcon(i) == [AES(first), AES(0), AES(0), AES(0)]

    def test_is_power_of_x(self):
        for i in range(1, 30):
            assert aes.rcon(i)[0] == AES(0x02) ** (i - 1)

    @pytest.mark.parametrize("i", [0, -1])
    def test_rejects_index_below_one(self, i):
        with pytest.raises(ValueError):
            aes.rcon(i)

    def test_entries_are_field_elements(self):
        assert all(type(x) is AES for x in aes.rcon(1))


class TestKeyExpansion128:
    KEY = "2b7e151628aed2a6abf7158809cf4f3c"
    # FIPS-197 Appendix A.1, one line per round key (w[4k] .. w[4k+3])
    EXPECTED = [
        "2b7e1516 28aed2a6 abf71588 09cf4f3c",
        "a0fafe17 88542cb1 23a33939 2a6c7605",
        "f2c295f2 7a96b943 5935807a 7359f67f",
        "3d80477d 4716fe3e 1e237e44 6d7a883b",
        "ef44a541 a8525b7f b671253b db0bad00",
        "d4d1c6f8 7c839d87 caf2b8bc 11f915bc",
        "6d88a37a 110b3efd dbf98641 ca0093fd",
        "4e54f70e 5f5fc9f3 84a64fb2 4ea6dc4f",
        "ead27321 b58dbad2 312bf560 7f8d292f",
        "ac7766f3 19fadc21 28d12941 575c006e",
        "d014f9a8 c9ee2589 e13f0cc8 b6630ca6",
    ]

    def expanded(self):
        return aes.key_expansion(bytes.fromhex(self.KEY))

    def test_length(self):
        assert len(self.expanded()) == 44

    def test_first_words_are_the_key(self):
        assert [word_hex(w) for w in self.expanded()[:4]] == self.EXPECTED[0].split()

    def test_first_derived_word(self):
        # w[4] = w[0] + (SubWord(RotWord(w[3])) + Rcon[1])
        assert word_hex(self.expanded()[4]) == "a0fafe17"

    def test_last_word(self):
        assert word_hex(self.expanded()[43]) == "b6630ca6"

    @pytest.mark.parametrize("k", range(11))
    def test_round_key(self, k):
        # Round key k is words[4k : 4k + 4]. A failure names the first wrong round.
        words = self.expanded()
        assert [word_hex(w) for w in words[4 * k: 4 * k + 4]] == self.EXPECTED[k].split()

    def test_appendix_c1_last_round_key(self):
        words = aes.key_expansion(bytes(range(16)))
        assert [word_hex(w) for w in words[40:]] == ["13111d7f", "e3944a17", "f307a78b", "4d2b30c5"]

    def test_round_key_1_works_with_add_round_key(self):
        # Round key 1 is the one TestAddRoundKey used.
        words = self.expanded()
        out = aes.add_round_key(state_from_hex(TestAddRoundKey.ROUND1_MIXED), words[4:8])
        assert hex_of(out) == TestAddRoundKey.ROUND2_START

    def test_shape_and_types(self):
        words = self.expanded()
        assert all(len(w) == 4 for w in words)
        assert all(type(x) is AES for w in words for x in w)

    def test_words_are_distinct_lists(self):
        # Shared lists would mean changing one word silently changes another.
        words = self.expanded()
        assert len({id(w) for w in words}) == 44

    def test_accepts_bytearray(self):
        assert aes.key_expansion(bytearray.fromhex(self.KEY)) == self.expanded()

    def test_one_key_bit_changes_last_round_key(self):
        # The schedule spreads each key bit: flipping one changes the final round key.
        key = bytearray.fromhex(self.KEY)
        key[0] ^= 0x01
        assert aes.key_expansion(bytes(key))[40:] != self.expanded()[40:]


# =============================================================================
# Step 7: Encryption  (FIPS-197 section 5.1, Algorithm 1; Appendix B, C.1)
# =============================================================================

class TestEncrypt128:
    B_INPUT = "3243f6a8885a308d313198a2e0370734"
    B_KEY = "2b7e151628aed2a6abf7158809cf4f3c"
    B_OUTPUT = "3925841d02dc09fbdc118597196a0b32"

    # Debugging aid, not used by the tests: FIPS-197 Appendix B "Start of Round" state
    # for rounds 1..10 (bytes in state order). If encrypt_block fails, print your state
    # at the start of each round and compare with these to find the first wrong round.
    B_ROUND_STARTS = [
        "193de3bea0f4e22b9ac68d2ae9f84808",
        "a49c7ff2689f352b6b5bea43026a5049",
        "aa8f5f0361dde3ef82d24ad26832469a",
        "486c4eee671d9d0d4de3b138d65f58e7",
        "e0927fe8c86363c0d9b1355085b8be01",
        "f1006f55c1924cef7cc88b325db5d50c",
        "260e2e173d41b77de86472a9fdd28b25",
        "5a4142b11949dc1fa3e019657a8c040c",
        "ea835cf00445332d655d98ad8596b0c5",
        "eb40f21e592e38848ba113e71bc342d2",
    ]

    # What you get if the final round wrongly includes MixColumns.
    B_OUTPUT_IF_MIX_IN_LAST_ROUND = "82ad1fb54c4712353f81f7967b430371"

    C1_INPUT = "00112233445566778899aabbccddeeff"
    C1_KEY = "000102030405060708090a0b0c0d0e0f"
    C1_OUTPUT = "69c4e0d86a7b0430d8cdb78070b4c55a"

    def encrypt_hex(self, block_hex, key_hex):
        return aes.encrypt_block(bytes.fromhex(block_hex), bytes.fromhex(key_hex)).hex()

    def test_final_round_has_no_mix_columns(self):
        out = self.encrypt_hex(self.B_INPUT, self.B_KEY)
        assert out != self.B_OUTPUT_IF_MIX_IN_LAST_ROUND, "the final round must skip MixColumns"

    def test_appendix_b(self):
        assert self.encrypt_hex(self.B_INPUT, self.B_KEY) == self.B_OUTPUT

    def test_appendix_c1(self):
        assert self.encrypt_hex(self.C1_INPUT, self.C1_KEY) == self.C1_OUTPUT

    def test_returns_16_bytes(self):
        out = aes.encrypt_block(bytes.fromhex(self.B_INPUT), bytes.fromhex(self.B_KEY))
        assert type(out) is bytes
        assert len(out) == 16

    def test_accepts_bytearray(self):
        out = aes.encrypt_block(bytearray.fromhex(self.B_INPUT), bytearray.fromhex(self.B_KEY))
        assert out.hex() == self.B_OUTPUT

    @pytest.mark.parametrize("length", [0, 15, 17, 32])
    def test_rejects_wrong_block_length(self, length):
        with pytest.raises(ValueError):
            aes.encrypt_block(bytes(length), bytes.fromhex(self.B_KEY))

    def test_deterministic(self):
        assert self.encrypt_hex(self.B_INPUT, self.B_KEY) == self.encrypt_hex(self.B_INPUT, self.B_KEY)

    def test_output_differs_from_input(self):
        rng = random.Random(11)
        for _ in range(20):
            block, key = random_block(rng), random_block(rng)
            assert aes.encrypt_block(block, key) != block

    def test_is_a_permutation_for_fixed_key(self):
        # Different blocks under one key never collide (AES is a bijection per key).
        rng = random.Random(12)
        key = random_block(rng)
        blocks = {random_block(rng) for _ in range(150)}
        assert len({aes.encrypt_block(b, key) for b in blocks}) == len(blocks)

    def test_avalanche_plaintext(self):
        # Flipping one input bit flips about half (64) of the 128 output bits on average.
        rng = random.Random(13)
        key = random_block(rng)
        total = 0
        for _ in range(100):
            block = bytearray(random_block(rng))
            bit = rng.randrange(128)
            a = aes.encrypt_block(bytes(block), key)
            block[bit // 8] ^= 1 << (bit % 8)
            b = aes.encrypt_block(bytes(block), key)
            total += sum(bin(x ^ y).count("1") for x, y in zip(a, b))
        assert 54 < total / 100 < 74

    def test_avalanche_key(self):
        rng = random.Random(14)
        block = random_block(rng)
        total = 0
        for _ in range(100):
            key = bytearray(random_block(rng))
            bit = rng.randrange(128)
            a = aes.encrypt_block(block, bytes(key))
            key[bit // 8] ^= 1 << (bit % 8)
            b = aes.encrypt_block(block, bytes(key))
            total += sum(bin(x ^ y).count("1") for x, y in zip(a, b))
        assert 54 < total / 100 < 74


# =============================================================================
# Step 8: Decryption  (FIPS-197 section 5.3, Algorithm 3; Appendix B, C.1)
# =============================================================================

class TestDecrypt128:
    B = TestEncrypt128

    # Debugging aid, not used by the tests: in the inverse cipher, the state right after
    # each InvSubBytes equals the encryption's "Start of Round" states in reverse order
    # (round 10 first, round 1 last). Print yours there and compare.
    B_STATES_AFTER_INV_SUB_BYTES = TestEncrypt128.B_ROUND_STARTS[::-1]

    # What Appendix B decrypts to with two common mistakes.
    B_OUTPUT_IF_KEYS_IN_FORWARD_ORDER = "57c2c49254271f375409c2a51bcbab98"
    B_OUTPUT_IF_INV_MIX_BEFORE_ADD_KEY = "9095d6f2690e14cc56236b56acbbfed7"

    def decrypt_hex(self, block_hex, key_hex):
        return aes.decrypt_block(bytes.fromhex(block_hex), bytes.fromhex(key_hex)).hex()

    def test_round_keys_used_backwards(self):
        out = self.decrypt_hex(self.B.B_OUTPUT, self.B.B_KEY)
        assert out != self.B_OUTPUT_IF_KEYS_IN_FORWARD_ORDER, "round keys must go from Nr down to 0"

    def test_add_round_key_before_inv_mix_columns(self):
        out = self.decrypt_hex(self.B.B_OUTPUT, self.B.B_KEY)
        assert out != self.B_OUTPUT_IF_INV_MIX_BEFORE_ADD_KEY, \
            "in Algorithm 3, AddRoundKey comes before InvMixColumns in each round"

    def test_appendix_b(self):
        assert self.decrypt_hex(self.B.B_OUTPUT, self.B.B_KEY) == self.B.B_INPUT

    def test_appendix_c1(self):
        assert self.decrypt_hex(self.B.C1_OUTPUT, self.B.C1_KEY) == self.B.C1_INPUT

    def test_returns_16_bytes(self):
        out = aes.decrypt_block(bytes.fromhex(self.B.B_OUTPUT), bytes.fromhex(self.B.B_KEY))
        assert type(out) is bytes
        assert len(out) == 16

    def test_accepts_bytearray(self):
        out = aes.decrypt_block(bytearray.fromhex(self.B.B_OUTPUT), bytearray.fromhex(self.B.B_KEY))
        assert out.hex() == self.B.B_INPUT

    @pytest.mark.parametrize("length", [0, 15, 17, 32])
    def test_rejects_wrong_block_length(self, length):
        with pytest.raises(ValueError):
            aes.decrypt_block(bytes(length), bytes.fromhex(self.B.B_KEY))

    def test_decrypt_undoes_encrypt(self):
        rng = random.Random(15)
        for _ in range(50):
            block, key = random_block(rng), random_block(rng)
            assert aes.decrypt_block(aes.encrypt_block(block, key), key) == block

    def test_encrypt_undoes_decrypt(self):
        # AES is a permutation, so this direction holds too.
        rng = random.Random(16)
        for _ in range(50):
            block, key = random_block(rng), random_block(rng)
            assert aes.encrypt_block(aes.decrypt_block(block, key), key) == block

    def test_wrong_key_gives_wrong_plaintext(self):
        rng = random.Random(17)
        block, key = random_block(rng), bytearray(random_block(rng))
        ct = aes.encrypt_block(block, bytes(key))
        key[0] ^= 0x01
        assert aes.decrypt_block(ct, bytes(key)) != block


# =============================================================================
# Step 9: AES-192 and AES-256  (FIPS-197 section 5.2, Appendix A.2, A.3, C.2, C.3)
# =============================================================================

C_INPUT = "00112233445566778899aabbccddeeff"


class TestKeyExpansion192And256:
    A2_KEY = "8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b"
    A3_KEY = "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4"

    @pytest.mark.parametrize("key_len, words", [(16, 44), (24, 52), (32, 60)])
    def test_length(self, key_len, words):
        # Nb * (Nr + 1) with Nr = 10, 12, 14
        assert len(aes.key_expansion(bytes(key_len))) == words

    def test_192_first_words_are_the_key(self):
        words = aes.key_expansion(bytes.fromhex(self.A2_KEY))
        assert "".join(word_hex(w) for w in words[:6]) == self.A2_KEY

    def test_192_first_derived_word(self):
        # Nk = 6: the first RotWord/SubWord/Rcon happens at i = 6, not i = 4.
        assert word_hex(aes.key_expansion(bytes.fromhex(self.A2_KEY))[6]) == "fe0c91f7"

    def test_192_last_round_key(self):
        words = aes.key_expansion(bytes.fromhex(self.A2_KEY))
        assert [word_hex(w) for w in words[48:]] == ["e98ba06f", "448c773c", "8ecc7204", "01002202"]

    def test_256_first_words_are_the_key(self):
        words = aes.key_expansion(bytes.fromhex(self.A3_KEY))
        assert "".join(word_hex(w) for w in words[:8]) == self.A3_KEY

    def test_256_first_derived_word(self):
        assert word_hex(aes.key_expansion(bytes.fromhex(self.A3_KEY))[8]) == "9ba35411"

    def test_256_extra_sub_word(self):
        # i = 12 has i mod Nk == 4: the extra SubWord rule (only when Nk > 6).
        assert word_hex(aes.key_expansion(bytes.fromhex(self.A3_KEY))[12]) == "a8b09c1a"

    def test_256_last_round_key(self):
        words = aes.key_expansion(bytes.fromhex(self.A3_KEY))
        assert [word_hex(w) for w in words[56:]] == ["fe4890d1", "e6188d0b", "046df344", "706c631e"]


class TestEncryptDecrypt192And256:
    C2_KEY = bytes(range(24)).hex()
    C2_OUTPUT = "dda97ca4864cdfe06eaf70a0ec0d7191"
    C3_KEY = bytes(range(32)).hex()
    C3_OUTPUT = "8ea2b7ca516745bfeafc49904b496089"

    # Wrong outputs from common mistakes.
    C2_OUTPUT_IF_10_ROUNDS = "56e16826eaf7401d4ddc1e4c3280698f"
    C3_OUTPUT_IF_10_ROUNDS = "aa3c2ec11893658550ebe95037725421"
    C3_OUTPUT_WITHOUT_EXTRA_SUB_WORD = "ebbb61b18794d6e67d2c1cf8eec9793e"

    def encrypt_hex(self, key_hex):
        return aes.encrypt_block(bytes.fromhex(C_INPUT), bytes.fromhex(key_hex)).hex()

    def decrypt_hex(self, ct_hex, key_hex):
        return aes.decrypt_block(bytes.fromhex(ct_hex), bytes.fromhex(key_hex)).hex()

    def test_192_round_count(self):
        assert self.encrypt_hex(self.C2_KEY) != self.C2_OUTPUT_IF_10_ROUNDS, "AES-192 has Nr = 12 rounds"

    def test_256_round_count(self):
        assert self.encrypt_hex(self.C3_KEY) != self.C3_OUTPUT_IF_10_ROUNDS, "AES-256 has Nr = 14 rounds"

    def test_256_extra_sub_word_in_cipher(self):
        assert self.encrypt_hex(self.C3_KEY) != self.C3_OUTPUT_WITHOUT_EXTRA_SUB_WORD, \
            "AES-256 key expansion needs SubWord when i mod Nk == 4"

    def test_encrypt_192(self):
        assert self.encrypt_hex(self.C2_KEY) == self.C2_OUTPUT

    def test_encrypt_256(self):
        assert self.encrypt_hex(self.C3_KEY) == self.C3_OUTPUT

    def test_decrypt_192(self):
        assert self.decrypt_hex(self.C2_OUTPUT, self.C2_KEY) == C_INPUT

    def test_decrypt_256(self):
        assert self.decrypt_hex(self.C3_OUTPUT, self.C3_KEY) == C_INPUT

    @pytest.mark.parametrize("key_len", [24, 32])
    def test_round_trip_random(self, key_len):
        rng = random.Random(key_len)
        for _ in range(30):
            block = random_block(rng)
            key = bytes(rng.randrange(256) for _ in range(key_len))
            assert aes.decrypt_block(aes.encrypt_block(block, key), key) == block

    def test_key_lengths_give_different_ciphers(self):
        # The same 16 leading key bytes, padded with zeros, must not give the same result.
        block = bytes.fromhex(C_INPUT)
        outs = {aes.encrypt_block(block, bytes(range(16)) + bytes(n)) for n in (0, 8, 16)}
        assert len(outs) == 3


class TestKeyLengthValidation:
    @pytest.mark.parametrize("key_len", [0, 4, 8, 15, 17, 20, 23, 25, 28, 31, 33, 64])
    def test_key_expansion_rejects(self, key_len):
        with pytest.raises(ValueError):
            aes.key_expansion(bytes(key_len))

    @pytest.mark.parametrize("key_len", [0, 15, 20, 28, 33])
    def test_encrypt_rejects(self, key_len):
        with pytest.raises(ValueError):
            aes.encrypt_block(bytes(16), bytes(key_len))

    @pytest.mark.parametrize("key_len", [0, 15, 20, 28, 33])
    def test_decrypt_rejects(self, key_len):
        with pytest.raises(ValueError):
            aes.decrypt_block(bytes(16), bytes(key_len))

    @pytest.mark.parametrize("key_len", [16, 24, 32])
    def test_accepts_valid_lengths(self, key_len):
        key = bytes(key_len)
        assert len(aes.decrypt_block(aes.encrypt_block(bytes(16), key), key)) == 16
