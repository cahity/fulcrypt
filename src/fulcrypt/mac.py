"""Message authentication codes: HMAC-SHA256 (FIPS 198-1, RFC 2104).

Design decisions:
- Built on fulcrypt.hashes.sha256, not hashlib, so it runs on our own SHA-256.
- Tags are compared with constant_time_equal, never ==: == stops at the first
  differing byte, which leaks how much of a forged tag was right.
"""

from fulcrypt.hashes import SHA256_BLOCK_SIZE, sha256
from fulcrypt.modes import xor_bytes


def hmac_sha256(key: bytes, message: bytes) -> bytes:
    """Return the 32-byte HMAC-SHA256 tag of message under key (any key length)."""
    if len(key) > SHA256_BLOCK_SIZE:
        key = sha256(key)
    if len(key) < SHA256_BLOCK_SIZE:
        key = key + ((SHA256_BLOCK_SIZE - len(key)) * b"\x00")

    ipad = b"\x36" * SHA256_BLOCK_SIZE
    opad = b"\x5c" * SHA256_BLOCK_SIZE

    k_in = xor_bytes(ipad, key)
    k_out = xor_bytes(opad, key)
    hmac = sha256(k_out + sha256(k_in + message))

    return hmac


def constant_time_equal(a: bytes, b: bytes) -> bool:
    """Compare two byte strings, looking at every byte whatever the differences."""
    if len(a) != len(b):
        return False

    res = 0
    for ab, bb in zip(a, b, strict=True):
        res |= ab ^ bb

    return not res


def hmac_sha256_verify(key: bytes, message: bytes, tag: bytes) -> bool:
    """Check an HMAC-SHA256 tag in constant time; False for any wrong or wrong-length tag."""
    hmac = hmac_sha256(key, message)
    return constant_time_equal(tag, hmac)
