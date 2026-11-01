"""Tests for SHA-3 (FIPS 202): Keccak constants, step mappings, Keccak-f[1600], sponge.

State: 25 lanes (64-bit ints), lane (x, y) at index x + 5*y. Expected states come from an
independent reference implementation whose SHA-3/SHAKE outputs match hashlib for every length.
"""

import hashlib

import pytest

from fulcrypt import hashes, sha3

M64 = (1 << 64) - 1


def idx(x, y):
    return x + 5 * y


KECCAK_RC_FIPS = (
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
)


KECCAK_ROT_FIPS = (  # [x][y]
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)


S = [
    0x0123456789ABCDEF,
    0x02468ACF13579BDE,
    0x0369D0369D0369CD,
    0x048D159E26AF37BC,
    0x05B05B05B05B05AB,
    0x06D3A06D3A06D39A,
    0x07F6E5D4C3B2A189,
    0x091A2B3C4D5E6F78,
    0x0A3D70A3D70A3D67,
    0x0B60B60B60B60B56,
    0x0C83FB72EA61D945,
    0x0DA740DA740DA734,
    0x0ECA8641FDB97523,
    0x0FEDCBA987654312,
    0x1111111111111101,
    0x123456789ABCDEF0,
    0x13579BE02468ACDF,
    0x147AE147AE147ACE,
    0x159E26AF37C048BD,
    0x16C16C16C16C16AC,
    0x17E4B17E4B17E49B,
    0x1907F6E5D4C3B28A,
    0x1A2B3C4D5E6F8079,
    0x1B4E81B4E81B4E68,
    0x1C71C71C71C71C57,
]
S_THETA = [
    0x10DC97FA507BDE84,
    0x19343332200F74C7,
    0x1E3481ED9B7674C3,
    0x24871BF5199E9493,
    0x177AA077098FB001,
    0x172C72F0E3D6C0F1,
    0x1C845C29F0EA4E90,
    0x14477AE74B2B7276,
    0x2A377EC8E83B9E48,
    0x19AA4D79D962BEFC,
    0x1D7C29EF33B1CA2E,
    0x16D5F9274755482D,
    0x1397D79AFBCC682D,
    0x2FE7C5C2B854E03D,
    0x03DBEA63A8C5A4AB,
    0x03CB84E5436CCD9B,
    0x0825221D173043C6,
    0x0927B09CA86167C0,
    0x359428C408F1EB92,
    0x040B976478B8A306,
    0x061B63E392C7F7F0,
    0x02754F18E79B5D93,
    0x07766D96581A9D77,
    0x3B448FDFD72AED47,
    0x0EBB3C6EC813A9FD,
]
S_RHO = [
    0x0123456789ABCDEF,
    0x048D159E26AF37BC,
    0x40DA740DA740DA73,
    0xE26AF37BC048D159,
    0x2D82D82D582D82D8,
    0xA06D39A06D3A06D3,
    0x2A18907F6E5D4C3B,
    0x468ACF13579BDE02,
    0xB3851EB851EB851E,
    0x60B60B60B560B60B,
    0x641FDB97530ECA28,
    0x9D0369D0369CD036,
    0xCBA9187654320FED,
    0x530ECA86241FDB97,
    0x8888808888888888,
    0x79BDE02468ACF135,
    0x159BE26AF37C048D,
    0x70A3D70A3D670A3D,
    0xD5E6F80917A2B3C4,
    0xC16C16C16C16AC16,
    0xC5F92C5F926C5F92,
    0x641FDB97530ECA28,
    0x23456789ABCDF00F,
    0x681B4E81B4E81B4E,
    0x71C71C71C715C71C,
]
S_PI = [
    0x0123456789ABCDEF,
    0x07F6E5D4C3B2A189,
    0x0ECA8641FDB97523,
    0x159E26AF37C048BD,
    0x1C71C71C71C71C57,
    0x048D159E26AF37BC,
    0x0B60B60B60B60B56,
    0x0C83FB72EA61D945,
    0x13579BE02468ACDF,
    0x1A2B3C4D5E6F8079,
    0x02468ACF13579BDE,
    0x091A2B3C4D5E6F78,
    0x0FEDCBA987654312,
    0x16C16C16C16C16AC,
    0x17E4B17E4B17E49B,
    0x05B05B05B05B05AB,
    0x06D3A06D3A06D39A,
    0x0DA740DA740DA734,
    0x147AE147AE147ACE,
    0x1B4E81B4E81B4E68,
    0x0369D0369D0369CD,
    0x0A3D70A3D70A3D67,
    0x1111111111111101,
    0x123456789ABCDEF0,
    0x1907F6E5D4C3B28A,
]
S_CHI = [
    0x000A155705ABADEE,
    0x06C28F4731FB8DEE,
    0x02599A370D5369CE,
    0x048E11FC2F0FFFF8,
    0x07F4D18DA20F17BB,
    0x0EDBAA45364A9DEA,
    0x05D3B55751B2B18E,
    0x085AAD346DEA6D68,
    0x0EAE70C7CD0AEDEF,
    0x0A44F39BA1062B57,
    0x0ECB7D7363D18946,
    0x0C8209727649A524,
    0x1EDA9651EDA96522,
    0x036F21CB6D058B56,
    0x10351199051D3731,
    0x161C367F10A88CF0,
    0x12D39D4835A8ACEE,
    0x163BA9576E386CCE,
    0x15AA34C72D5080ED,
    0x1782E596E52C36A3,
    0x15CCB976413BE4EA,
    0x1843775574D3FC8A,
    0x1E1A7A454FAB906E,
    0x18CAB1D6E20BAEE0,
    0x1472819DE5070E57,
]
S_ROUND0 = [
    0x53DCFFEEEA479E55,
    0xBC6917E90580BF0F,
    0xA45BC89C368596F4,
    0x08451BC762462E85,
    0x6B3AB204EFFF42A4,
    0x79F9A019224630EF,
    0xD785162FCF803822,
    0x0B6743CBD68E0078,
    0x1969E14D940B82F7,
    0x66EADB900682D9AE,
    0xB6486664701DAB84,
    0x1B59FDEA4A7C9D81,
    0x0178A2C73D1FD7E2,
    0x3BF74018B8BDE786,
    0x8E18D28D55000C6C,
    0xFACCED8C0D9B4552,
    0xB5664F31D0072F8E,
    0x50D59D9219379E37,
    0x600A6D30B3C8D191,
    0x421B469EAD9300E1,
    0x854F647BEF5CBD20,
    0xBD1C39B9767CD7CD,
    0x62965DE1E591059C,
    0x1F93361FF79943B6,
    0x29C527E79E4D7683,
]
S_F1600 = [
    0x417587A73FF14736,
    0x873BBA4514A8A528,
    0x05A18DBC335777EC,
    0xA41CC61DDF026A5B,
    0xB0E172792D3CC4E5,
    0x472C635570B62583,
    0xE2D65383463F6F17,
    0x60BAA6F9379C7BA9,
    0xA2C422CB35DDC877,
    0x510CE38FEB44910D,
    0x924C9EF2011FF0BC,
    0x0DF1614C89E62A0D,
    0xC149908C2F31B8B7,
    0xE4D870B996ED8882,
    0xBCC0333872CEFEBE,
    0x62A75A1066987052,
    0xBB6FFC6B1724AC28,
    0xD67DD7B4B493A612,
    0x9E15E34A81636C19,
    0xBB0F092AE5A26D76,
    0x1B27912AF146FC4A,
    0xC55C3E7DE26F2FEC,
    0x704808FC5A6582DD,
    0x3788B80BA50A9FDA,
    0x5E9F70E888A1388F,
]
ZERO_F1600 = [
    0xF1258F7940E1DDE7,
    0x84D5CCF933C0478A,
    0xD598261EA65AA9EE,
    0xBD1547306F80494D,
    0x8B284E056253D057,
    0xFF97A42D7F8E6FD4,
    0x90FEE5A0A44647C4,
    0x8C5BDA0CD6192E76,
    0xAD30A6F71B19059C,
    0x30935AB7D08FFC64,
    0xEB5AA93F2317D635,
    0xA9A6E6260D712103,
    0x81A57C16DBCF555F,
    0x43B831CD0347C826,
    0x01F22F1A11A5569F,
    0x05E5635A21D9AE61,
    0x64BEFEF28CC970F2,
    0x613670957BC46611,
    0xB87C5A554FD00ECB,
    0x8C3EE88A1CCF32C8,
    0x940C7922AE3A2614,
    0x1841F924A2C509E4,
    0x16F53526E70465C2,
    0x75F644E97F30A13B,
    0xEAF1FF7B5CECA249,
]


# =============================================================================
# Step 16: Keccak constants  (FIPS 202 sections 3.2.2 and 3.2.5)
# =============================================================================


class TestKeccakConstants:
    def test_rounds(self):
        assert sha3.KECCAK_ROUNDS == 24

    def test_round_constants(self):
        assert sha3.KECCAK_RC == KECCAK_RC_FIPS

    def test_round_constants_function(self):
        rc = sha3.keccak_round_constants()
        assert type(rc) is tuple and len(rc) == 24
        assert rc == KECCAK_RC_FIPS

    def test_round_constants_only_use_bits_2j_minus_1(self):
        # RC[i] can only have bits 0, 1, 3, 7, 15, 31, 63 set.
        allowed = sum(1 << ((1 << j) - 1) for j in range(7))
        assert all(rc & ~allowed == 0 for rc in sha3.KECCAK_RC)

    def test_rotation_offsets(self):
        assert sha3.KECCAK_ROT == KECCAK_ROT_FIPS

    def test_rotation_offsets_function(self):
        rot = sha3.keccak_rotation_offsets()
        assert type(rot) is tuple and len(rot) == 5
        assert all(type(row) is tuple and len(row) == 5 for row in rot)
        assert rot == KECCAK_ROT_FIPS

    def test_rotation_offsets_indexed_x_then_y(self):
        # Lane (1, 0) rotates by 1, lane (0, 1) by 36: an [x][y] vs [y][x] mix-up swaps them.
        assert sha3.KECCAK_ROT[1][0] == 1
        assert sha3.KECCAK_ROT[0][1] == 36
        assert sha3.KECCAK_ROT[0][0] == 0

    def test_rotation_offsets_are_distinct(self):
        # The 24 non-zero offsets mod 64 are all different (lane (0, 0) stays 0).
        offsets = [sha3.KECCAK_ROT[x][y] for x in range(5) for y in range(5) if (x, y) != (0, 0)]
        assert len(set(offsets)) == 24


# =============================================================================
# Step 17: Step mappings and Keccak-f[1600]  (FIPS 202 sections 3.2, 3.3)
# =============================================================================


def single_bit(x, y, bit=0):
    a = [0] * 25
    a[idx(x, y)] = 1 << bit
    return a


class TestTheta:
    def test_known_state(self):
        assert sha3.theta(S) == S_THETA

    def test_zero(self):
        assert sha3.theta([0] * 25) == [0] * 25

    def test_single_bit_spreads_to_two_columns(self):
        # A bit in column x flips: itself, column x+1 (same bit) and column x-1 (bit + 1).
        out = sha3.theta(single_bit(2, 3, bit=5))
        assert out[idx(2, 3)] == 1 << 5
        for y in range(5):
            assert out[idx(3, y)] == 1 << 5
            assert out[idx(1, y)] == 1 << 6
            assert out[idx(0, y)] == out[idx(4, y)] == 0

    def test_rotation_wraps(self):
        # Bit 63 of column x+1's parity rotates into bit 0 of column x's effect.
        out = sha3.theta(single_bit(3, 0, bit=63))
        assert out[idx(2, 4)] == 1

    def test_does_not_modify_input(self):
        a = list(S)
        sha3.theta(a)
        assert a == S


class TestRho:
    def test_known_state(self):
        assert sha3.rho(S) == S_RHO

    @pytest.mark.parametrize("x, y", [(0, 0), (1, 0), (0, 1), (4, 4), (2, 3)])
    def test_rotates_each_lane_by_its_offset(self, x, y):
        n = KECCAK_ROT_FIPS[x][y]
        assert sha3.rho(single_bit(x, y))[idx(x, y)] == 1 << n

    def test_top_bit_wraps(self):
        out = sha3.rho(single_bit(1, 0, bit=63))
        assert out[idx(1, 0)] == 1

    def test_does_not_modify_input(self):
        a = list(S)
        sha3.rho(a)
        assert a == S


class TestPi:
    def test_known_state(self):
        assert sha3.pi(S) == S_PI

    @pytest.mark.parametrize("x, y", [(0, 0), (1, 0), (0, 1), (3, 2), (4, 4)])
    def test_moves_lanes(self, x, y):
        # A'[x, y] = A[(x + 3y) mod 5, x]
        a = list(range(25))
        assert sha3.pi(a)[idx(x, y)] == a[idx((x + 3 * y) % 5, x)]

    def test_is_a_permutation(self):
        assert sorted(sha3.pi(list(range(25)))) == list(range(25))

    def test_lane_0_0_stays(self):
        assert sha3.pi(list(range(25)))[0] == 0

    def test_does_not_modify_input(self):
        a = list(S)
        sha3.pi(a)
        assert a == S


class TestChi:
    def test_known_state(self):
        assert sha3.chi(S) == S_CHI

    def test_rule(self):
        # A'[x] = A[x] ^ (~A[x+1] & A[x+2]) along each row, on 64-bit lanes.
        a = [0] * 25
        a[idx(1, 0)], a[idx(2, 0)] = 0, M64  # ~A[1] & A[2] = all ones -> lane (0, 0) flips
        assert sha3.chi(a)[idx(0, 0)] == M64

    def test_stays_64_bit(self):
        # ~ on a Python int is negative; the result must still be a 64-bit lane.
        out = sha3.chi(list(S))
        assert all(0 <= lane <= M64 for lane in out)

    def test_rows_are_independent(self):
        a = [0] * 25
        a[idx(2, 3)] = M64
        out = sha3.chi(a)
        assert all(out[i] == 0 for i in range(25) if i // 5 != 3)

    def test_does_not_modify_input(self):
        a = list(S)
        sha3.chi(a)
        assert a == S


class TestIota:
    @pytest.mark.parametrize("ir", [0, 1, 5, 23])
    def test_xors_round_constant_into_lane_0(self, ir):
        out = sha3.iota(S, ir)
        assert out[0] == S[0] ^ KECCAK_RC_FIPS[ir]
        assert out[1:] == S[1:]

    def test_does_not_modify_input(self):
        a = list(S)
        sha3.iota(a, 3)
        assert a == S


class TestKeccakF1600:
    def test_zero_state(self):
        # Published Keccak-f[1600] test: the permutation of the all-zero state.
        out = sha3.keccak_f1600([0] * 25)
        assert out[0] == 0xF1258F7940E1DDE7
        assert out == ZERO_F1600

    def test_known_state(self):
        assert sha3.keccak_f1600(S) == S_F1600

    def test_one_round_order(self):
        # One round is iota(chi(pi(rho(theta(a))))), in that order.
        assert sha3.iota(sha3.chi(sha3.pi(sha3.rho(sha3.theta(S)))), 0) == S_ROUND0

    def test_lanes_stay_64_bit(self):
        assert all(0 <= lane <= M64 for lane in sha3.keccak_f1600(S))

    def test_does_not_modify_input(self):
        a = list(S)
        sha3.keccak_f1600(a)
        assert a == S

    @pytest.mark.parametrize("n", [0, 24, 26])
    def test_rejects_wrong_lane_count(self, n):
        with pytest.raises(ValueError):
            sha3.keccak_f1600([0] * n)


# =============================================================================
# Step 18: Sponge, SHA-3 and SHAKE  (FIPS 202 sections 4, 5, 6)
# =============================================================================


def msg(n):
    return bytes((7 * i + 3) % 256 for i in range(n))


SHA3 = [
    ("sha3_224", hashlib.sha3_224),
    ("sha3_256", hashlib.sha3_256),
    ("sha3_384", hashlib.sha3_384),
    ("sha3_512", hashlib.sha3_512),
]


class TestSha3:
    def test_abc(self):
        # FIPS 202 example (NIST CSRC): SHA3-256("abc").
        expected = "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
        assert sha3.sha3_256(b"abc").hex() == expected

    def test_empty(self):
        expected = "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        assert sha3.sha3_256(b"").hex() == expected

    @pytest.mark.parametrize("name, reference", SHA3)
    def test_matches_hashlib_every_length(self, name, reference):
        # 0..300 bytes covers the one-byte-padding case (0x86) and multi-block messages
        # for every rate (144, 136, 104, 72 bytes).
        fn = getattr(sha3, name)
        for n in range(0, 301):
            assert fn(msg(n)) == reference(msg(n)).digest(), n

    @pytest.mark.parametrize(
        "name, size", [("sha3_224", 28), ("sha3_256", 32), ("sha3_384", 48), ("sha3_512", 64)]
    )
    def test_digest_size(self, name, size):
        out = getattr(sha3, name)(b"abc")
        assert type(out) is bytes and len(out) == size

    def test_accepts_bytearray(self):
        assert sha3.sha3_256(bytearray(b"abc")) == sha3.sha3_256(b"abc")


class TestShake:
    @pytest.mark.parametrize("length", [0, 1, 16, 32, 167, 168, 169, 400])
    def test_shake128_lengths(self, length):
        # Lengths around the 168-byte rate need one more permutation while squeezing.
        assert sha3.shake128(b"abc", length) == hashlib.shake_128(b"abc").digest(length)

    @pytest.mark.parametrize("length", [0, 32, 135, 136, 137, 300])
    def test_shake256_lengths(self, length):
        assert sha3.shake256(b"abc", length) == hashlib.shake_256(b"abc").digest(length)

    def test_matches_hashlib_every_message_length(self):
        for n in range(0, 200):
            assert sha3.shake128(msg(n), 40) == hashlib.shake_128(msg(n)).digest(40), n

    def test_shorter_output_is_a_prefix(self):
        long = sha3.shake128(b"seed", 500)
        for length in (0, 1, 168, 300):
            assert sha3.shake128(b"seed", length) == long[:length]

    def test_domain_separation(self):
        # Same rate (136), different suffix: SHAKE256 and SHA3-256 differ.
        assert sha3.shake256(b"abc", 32) != sha3.sha3_256(b"abc")


class TestKeccakSponge:
    def test_sha3_256_is_the_sponge(self):
        assert sha3.keccak_sponge(b"abc", 136, 0x06, 32) == sha3.sha3_256(b"abc")

    def test_one_free_byte_gets_0x86(self):
        # 135 bytes leave one byte in a 136-byte block: suffix 0x06 and final 0x80 share it.
        data = msg(135)
        block = data + b"\x86"
        state = [int.from_bytes(block[8 * i : 8 * i + 8], "little") for i in range(17)] + [0] * 8
        expected = sha3.keccak_f1600(state)
        out = b"".join(lane.to_bytes(8, "little") for lane in expected)[:32]
        assert sha3.keccak_sponge(data, 136, 0x06, 32) == out

    def test_lanes_are_little_endian(self):
        # Empty message: the first block is 06 00 .. 00 80; lane 0 is 0x06, not 0x06 << 56.
        state = [0] * 25
        state[0] = 0x06
        state[16] = 0x80 << 56
        out = b"".join(lane.to_bytes(8, "little") for lane in sha3.keccak_f1600(state))
        assert sha3.keccak_sponge(b"", 136, 0x06, 32) == out[:32]

    def test_no_length_extension(self):
        # Step 6's idea on SHA3-256: restart the sponge from the digest and absorb more.
        # It fails: the digest is only 4 of the 25 lanes; the other 21 (incl. the 64-byte
        # capacity) never leave the sponge, so the attacker restarts from the wrong state.
        secret, m, suffix = b"secret key", b"user=alice", b"&role=admin"
        known = secret + m  # only its length matters to the attacker
        digest = sha3.sha3_256(known)

        def pad(data):  # SHA3-256 padding of a message shorter than one 136-byte block
            block = bytearray(data + bytes(136 - len(data)))
            block[len(data)] ^= 0x06
            block[-1] ^= 0x80
            return bytes(block)

        glue = pad(known)[len(known) :]
        state = [int.from_bytes(digest[8 * i : 8 * i + 8], "little") for i in range(4)]
        state += [0] * 21
        block = pad(suffix)
        state = [
            s ^ int.from_bytes(block[8 * i : 8 * i + 8], "little") if i < 17 else s
            for i, s in enumerate(state)
        ]
        forged = b"".join(lane.to_bytes(8, "little") for lane in sha3.keccak_f1600(state))[:32]

        assert forged != sha3.sha3_256(known + glue + suffix)
        # The same idea works on SHA-256 (Step 6):
        ext, forged256 = hashes.sha256_extend(hashes.sha256(known), len(known), suffix)
        assert forged256 == hashes.sha256(known + ext)
