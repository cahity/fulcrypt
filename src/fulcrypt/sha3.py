"""SHA-3 and SHAKE (FIPS 202): the Keccak-f[1600] permutation inside a sponge.

Design decisions:
- The state is a list of 25 ints, one 64-bit lane each; lane (x, y) is at index x + 5*y.
  Every operation is masked with MASK_64, since Python ints never overflow.
- The constants (round constants, rotation offsets) are derived, not copied, and checked
  against the published tables in the tests.
- Each step mapping (theta, rho, pi, chi, iota) is a separate public function that returns
  a new state, so each can be tested on its own.
- Lanes are read and written little-endian, as FIPS 202 specifies; SHA-2 is big-endian.
- No length extension: the digest is only part of the state (4 of 25 lanes for SHA3-256),
  so it cannot be used to restart the sponge.
"""

# FIPS 202 parameters for Keccak-f[1600]; everything else is derived from l.
KECCAK_L = 6
KECCAK_LANE_BITS = 1 << KECCAK_L  # w = 64
KECCAK_LANE_SIZE = KECCAK_LANE_BITS // 8  # 8 bytes
KECCAK_LANES = 5 * 5
KECCAK_WIDTH = KECCAK_LANES * KECCAK_LANE_BITS  # b = 1600 bits
KECCAK_STATE_SIZE = KECCAK_WIDTH // 8  # 200 bytes
KECCAK_ROUNDS = 12 + 2 * KECCAK_L  # n_r = 24
MASK_64 = (1 << KECCAK_LANE_BITS) - 1

SHA3_SUFFIX = 0x06  # domain bits 01, then the first pad10*1 bit
SHAKE_SUFFIX = 0x1F  # domain bits 1111, then the first pad10*1 bit


def keccak_round_constants() -> tuple[int, ...]:
    """Derive the 24 iota round constants from the LFSR x^8 + x^6 + x^5 + x^4 + 1.

    Round ir sets bit 2^j - 1 of its constant to rc(j + 7*ir) for j = 0..6, so only
    bits 0, 1, 3, 7, 15, 31 and 63 can be set.
    """

    def rc(t: int) -> int:
        """Output bit t of the LFSR (the state starts as 1, 0, ..., 0)."""
        r = [1, 0, 0, 0, 0, 0, 0, 0]
        rem = t % 255
        for _ in range(rem):
            r = [0] + r
            r[0] ^= r[8]
            r[4] ^= r[8]
            r[5] ^= r[8]
            r[6] ^= r[8]
            r = r[:-1]
        return r[0]

    rc_list = [0] * KECCAK_ROUNDS
    for ir in range(KECCAK_ROUNDS):
        for j in range(7):
            bit = rc(j + 7 * ir)
            rc_list[ir] = rc_list[ir] | (bit << (2**j - 1))

    return tuple(rc_list)


def keccak_rotation_offsets() -> tuple[tuple[int, ...], ...]:
    """Derive the rho offsets, indexed [x][y]: lane (0, 0) stays 0.

    Start at (1, 0); step t gets (t+1)(t+2)/2 mod 64, then (x, y) moves to (y, 2x + 3y).
    The walk visits the other 24 lanes once each.
    """
    offset = [[0] * 5 for _ in range(5)]

    x, y = 1, 0
    for t in range(KECCAK_LANES - 1):  # every lane except (0, 0)
        offset[x][y] = (((t + 1) * (t + 2)) // 2) % KECCAK_LANE_BITS
        x, y = y, ((2 * x + 3 * y) % 5)
    return tuple(tuple(el) for el in offset)


KECCAK_RC = keccak_round_constants()
KECCAK_ROT = keccak_rotation_offsets()


def _left_rot_64(n, k):
    """Rotate a 64-bit lane left by k bits."""
    return ((n << k) | (n >> (KECCAK_LANE_BITS - k))) & MASK_64


def theta(a: list[int]) -> list[int]:
    """XOR each lane with the parities of the two neighbouring columns (one rotated by 1)."""
    c = [0] * 5
    for x in range(5):
        for i in range(5):
            c[x] ^= a[x + i * 5]
    d = [0] * 5
    for x in range(5):
        d[x] = c[(x - 1) % 5] ^ _left_rot_64(c[(x + 1) % 5], 1)

    res = a.copy()
    for x in range(5):
        for y in range(5):
            res[x + y * 5] ^= d[x]

    return res


def rho(a: list[int]) -> list[int]:
    """Rotate each lane (x, y) left by KECCAK_ROT[x][y]."""
    res = a.copy()
    for x in range(5):
        for y in range(5):
            res[x + y * 5] = _left_rot_64(res[x + y * 5], KECCAK_ROT[x][y])
    return res


def pi(a: list[int]) -> list[int]:
    """Move the lanes: A'[x, y] = A[(x + 3y) mod 5, x]."""
    res = a.copy()
    for x in range(5):
        for y in range(5):
            res[x + y * 5] = a[((x + y * 3) % 5) + x * 5]
    return res


def chi(a: list[int]) -> list[int]:
    """The only non-linear step: A[x, y] ^= ~A[x+1, y] & A[x+2, y], within each row."""
    res = a.copy()
    for x in range(5):
        for y in range(5):
            res[x + y * 5] = (
                res[x + y * 5] ^ (~a[((x + 1) % 5) + y * 5] & a[((x + 2) % 5) + y * 5])
            ) & MASK_64
    return res


def iota(a: list[int], ir: int) -> list[int]:
    """XOR round constant KECCAK_RC[ir] into lane (0, 0), so the rounds differ."""
    res = a.copy()
    res[0] ^= KECCAK_RC[ir]
    return res


def keccak_f1600(a: list[int]) -> list[int]:
    """Apply the 24 rounds of iota . chi . pi . rho . theta to a 25-lane state."""
    if len(a) != KECCAK_LANES:
        raise ValueError(f"State must be exactly {KECCAK_LANES} lanes, got {len(a)}.")

    for ir in range(KECCAK_ROUNDS):
        a = iota(chi(pi(rho(theta(a)))), ir)

    return a


def keccak_sponge(data: bytes, rate: int, suffix: int, length: int) -> bytes:
    """Keccak sponge: pad, absorb rate-byte blocks, squeeze length bytes.

    Padding: the domain suffix (0x06 for SHA-3, 0x1F for SHAKE) in the first padding byte,
    0x80 in the last one; they share a byte (0x86) when only one is free.
    The other KECCAK_STATE_SIZE - rate bytes (the capacity) are never output: the security.
    """
    zero_len = 1
    rem = (len(data) + zero_len) % rate
    if rem != 0:
        zero_len += rate - rem

    pad = (suffix & ((1 << 8) - 1)).to_bytes(1) + b"\x00" * (zero_len - 1)
    pad = pad[:-1] + (pad[-1] ^ 0x80).to_bytes(1)

    padded_data = bytes(data) + pad

    # Absorb
    state = [0] * KECCAK_LANES
    for i in range(0, len(padded_data), rate):
        block = padded_data[i : i + rate]
        for lane_i in range(rate // KECCAK_LANE_SIZE):
            j = lane_i * KECCAK_LANE_SIZE
            lane = int.from_bytes(block[j : j + KECCAK_LANE_SIZE], "little")
            state[lane_i] ^= lane
        state = keccak_f1600(state)

    # Squeeze
    sq_rounds = (length + rate - 1) // rate
    res = bytearray()
    for _ in range(sq_rounds):
        for lane_i in range(rate // KECCAK_LANE_SIZE):
            res += state[lane_i].to_bytes(KECCAK_LANE_SIZE, "little")
        state = keccak_f1600(state)

    return bytes(res[:length])


def keccak(data: bytes, capacity: int, suffix: int, length: int) -> bytes:
    """KECCAK[c] as in FIPS 202: the sponge with rate r = b - c; capacity in bits."""
    return keccak_sponge(data, (KECCAK_WIDTH - capacity) // 8, suffix, length)


def sha3_224(data: bytes) -> bytes:
    """Return the 28-byte SHA3-224 digest of data: KECCAK[448](M || 01, 224)."""
    return keccak(data, 2 * 224, SHA3_SUFFIX, 224 // 8)


def sha3_256(data: bytes) -> bytes:
    """Return the 32-byte SHA3-256 digest of data: KECCAK[512](M || 01, 256)."""
    return keccak(data, 2 * 256, SHA3_SUFFIX, 256 // 8)


def sha3_384(data: bytes) -> bytes:
    """Return the 48-byte SHA3-384 digest of data: KECCAK[768](M || 01, 384)."""
    return keccak(data, 2 * 384, SHA3_SUFFIX, 384 // 8)


def sha3_512(data: bytes) -> bytes:
    """Return the 64-byte SHA3-512 digest of data: KECCAK[1024](M || 01, 512)."""
    return keccak(data, 2 * 512, SHA3_SUFFIX, 512 // 8)


def shake128(data: bytes, length: int) -> bytes:
    """Return length bytes of SHAKE128 output: KECCAK[256](M || 1111, 8 * length)."""
    return keccak(data, 256, SHAKE_SUFFIX, length)


def shake256(data: bytes, length: int) -> bytes:
    """Return length bytes of SHAKE256 output: KECCAK[512](M || 1111, 8 * length)."""
    return keccak(data, 512, SHAKE_SUFFIX, length)
