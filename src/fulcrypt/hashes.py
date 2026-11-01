"""SHA-256 and SHA-224 hash functions (FIPS 180-4).

Design decisions:
- Words are plain Python ints. Python ints never overflow, so every addition and shift
  is masked with MASK_32 to stay within 32 bits.
- The constants (H0, K) are derived from the primes, not copied, and checked against
  the published values in the tests.
- Padding and compression are separate public functions, so each can be tested on its
  own and reused (SHA-224 shares both; the length extension attack restarts compression).
"""

import math

MASK_32 = (1 << 32) - 1
SHA256_WORD_SIZE = 4  # bytes per 32-bit word
SHA256_STATE_WORDS = 8
SHA256_ROUNDS = 64
SHA256_LENGTH_SIZE = 8  # the 64-bit message length at the end of the padding
SHA256_BLOCK_SIZE = 16 * SHA256_WORD_SIZE  # 64 bytes
SHA256_DIGEST_SIZE = SHA256_STATE_WORDS * SHA256_WORD_SIZE  # 32 bytes


def sha2_constants() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Derive SHA256_H0, SHA256_K and SHA224_H0 from the first 64 primes.

    SHA256_H0: first 32 bits of the fractional parts of the square roots of primes 1-8.
    SHA256_K: first 32 bits of the fractional parts of the cube roots of primes 1-64.
    SHA224_H0: second 32 bits of the fractional parts of the square roots of primes 9-16.
    """
    sieve = [True] * 312
    sieve[0] = False
    sieve[1] = False

    for p in range(312):
        if not sieve[p]:
            continue

        for pp in range(p + p, 312, p):
            sieve[pp] = False

    primes = [p for p in range(312) if sieve[p]]

    if len(primes) != SHA256_ROUNDS:
        raise RuntimeError(f"Expected {SHA256_ROUNDS} primes below 312, found {len(primes)}.")

    def icbrt(n: int) -> int:
        """Integer cube root: the largest m with m^3 <= n (binary search)."""
        lo, hi = 0, n

        while lo < hi:
            m = lo + (hi - lo + 1) // 2
            if m * m * m > n:
                hi = m - 1
            else:
                lo = m
        return lo

    h0: list[int] = []
    k: list[int] = []
    h0224: list[int] = []
    for p in primes:
        if len(h0) < SHA256_STATE_WORDS:
            h0.append(math.isqrt(p << 64) & MASK_32)
        k.append(icbrt(p << 96) & MASK_32)
        if 8 < len(k) < 17:
            h0224.append(math.isqrt(p << 128) & MASK_32)

    return tuple(h0), tuple(k), tuple(h0224)


SHA256_H0, SHA256_K, SHA224_H0 = sha2_constants()


def sha256_pad(message: bytes) -> bytes:
    """Pad a message to a multiple of 64 bytes: 0x80, zeros, then the 64-bit bit length."""
    length = (len(message) * 8).to_bytes(SHA256_LENGTH_SIZE, "big")
    rem = len(message) % SHA256_BLOCK_SIZE
    zero_len = SHA256_BLOCK_SIZE - rem - 1 - SHA256_LENGTH_SIZE  # 0x80 byte, length
    if zero_len < 0:
        zero_len += SHA256_BLOCK_SIZE

    return bytes(message) + b"\x80" + zero_len * b"\x00" + length


def sha256_compress(state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
    """Compress one 64-byte block into an 8-word state; return the new state."""
    if len(state) != SHA256_STATE_WORDS:
        raise ValueError(f"State must be exactly {SHA256_STATE_WORDS} words, got {len(state)}.")
    if len(block) != SHA256_BLOCK_SIZE:
        raise ValueError(f"Block must be exactly {SHA256_BLOCK_SIZE} bytes, got {len(block)}.")

    def right_rot(n, k):
        """Rotate a 32-bit word right by k bits."""
        return ((n >> k) | (n << (8 * SHA256_WORD_SIZE - k))) & MASK_32

    w = SHA256_WORD_SIZE
    words = [int.from_bytes(block[i : i + w], "big") for i in range(0, SHA256_BLOCK_SIZE, w)]
    while len(words) < SHA256_ROUNDS:
        n = len(words)
        s0 = right_rot(words[n - 15], 7) ^ right_rot(words[n - 15], 18) ^ (words[n - 15] >> 3)
        s1 = right_rot(words[n - 2], 17) ^ right_rot(words[n - 2], 19) ^ (words[n - 2] >> 10)
        new_word = (s1 + words[n - 7] + s0 + words[n - 16]) & MASK_32
        words.append(new_word)

    a, b, c, d, e, f, g, h = state

    for t in range(SHA256_ROUNDS):
        s1 = right_rot(e, 6) ^ right_rot(e, 11) ^ right_rot(e, 25)
        ch = (e & f) ^ ((~e) & g)
        temp1 = h + s1 + ch + SHA256_K[t] + words[t] & MASK_32
        s0 = right_rot(a, 2) ^ right_rot(a, 13) ^ right_rot(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = s0 + maj & MASK_32

        h = g
        g = f
        f = e
        e = (d + temp1) & MASK_32
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & MASK_32

    new_state = []
    for olds, news in zip(state, (a, b, c, d, e, f, g, h), strict=True):
        new_state.append((olds + news) & MASK_32)

    return tuple(new_state)


def sha256(data: bytes) -> bytes:
    """Return the 32-byte SHA-256 digest of data."""
    padded = sha256_pad(data)

    state = SHA256_H0
    for i in range(0, len(padded), SHA256_BLOCK_SIZE):
        state = sha256_compress(state, padded[i : i + SHA256_BLOCK_SIZE])

    return b"".join(n.to_bytes(SHA256_WORD_SIZE, "big") for n in state)


def sha224(data: bytes) -> bytes:
    """Return the 28-byte SHA-224 digest: SHA-256 with SHA224_H0, truncated to 7 words."""
    padded = sha256_pad(data)

    state = SHA224_H0
    for i in range(0, len(padded), SHA256_BLOCK_SIZE):
        state = sha256_compress(state, padded[i : i + SHA256_BLOCK_SIZE])

    return b"".join(n.to_bytes(SHA256_WORD_SIZE, "big") for n in state[:-1])


def sha256_extend(digest: bytes, original_length: int, suffix: bytes) -> tuple[bytes, bytes]:
    """Length extension attack: forge sha256(secret || m || glue || suffix) without the secret.

    Needs only digest = sha256(secret || m) and original_length = len(secret || m).
    Restarts the compression from the digest, since it is the state after the glue.
    Returns (glue || suffix, new_digest); glue is the original message's padding.
    """
    if len(digest) != SHA256_DIGEST_SIZE:
        raise ValueError(f"Digest must be exactly {SHA256_DIGEST_SIZE} bytes, got {len(digest)}.")
    if original_length < 0:
        raise ValueError(f"Original length must not be negative, got {original_length}.")

    glue = sha256_pad(b"\x00" * original_length)[original_length:]

    new_length = original_length + len(glue) + len(suffix)
    new_pad = sha256_pad(b"\x00" * new_length)[new_length:]

    extension = bytes(suffix) + new_pad

    w = SHA256_WORD_SIZE
    state = tuple(int.from_bytes(digest[i : i + w], "big") for i in range(0, SHA256_DIGEST_SIZE, w))
    for i in range(0, len(extension), SHA256_BLOCK_SIZE):
        state = sha256_compress(state, extension[i : i + SHA256_BLOCK_SIZE])

    return (glue + bytes(suffix), b"".join(n.to_bytes(SHA256_WORD_SIZE, "big") for n in state))
