"""Message authentication codes: HMAC-SHA256 (FIPS 198-1), CBC-MAC and CMAC (SP 800-38B).

Design decisions:
- Built on fulcrypt.hashes.sha256, not hashlib, so it runs on our own SHA-256.
- Tags are compared with constant_time_equal, never ==: == stops at the first
  differing byte, which leaks how much of a forged tag was right.
- CBC-MAC is here to show its weakness: it is secure only for messages of one fixed
  length. Tags of different-length messages can be combined into a forgery.
- CMAC fixes it by masking the last block with a secret subkey: K1 for a full block,
  K2 for a 10..0-padded one. Subkeys come from doubling L = AES(K, 0) in GF(2^128).
"""

from fulcrypt.aes import encrypt_block as aes_encrypt_block
from fulcrypt.hashes import SHA256_BLOCK_SIZE, sha256
from fulcrypt.modes import BLOCK_SIZE as AES_BLOCK_SIZE
from fulcrypt.modes import cbc_encrypt, xor_bytes


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


def cbc_mac(key: bytes, message: bytes) -> bytes:
    """Return the CBC-MAC tag: the last block of AES-CBC with a zero IV (non-empty, aligned)."""
    if len(message) == 0:
        raise ValueError(f"Message must be at least one {AES_BLOCK_SIZE}-byte block, got 0 bytes.")

    iv = AES_BLOCK_SIZE * b"\x00"
    ciphertext = cbc_encrypt(message, key, iv)
    return ciphertext[-AES_BLOCK_SIZE:]


def cmac_subkeys(key: bytes) -> tuple[bytes, bytes]:
    """Derive the CMAC subkeys: K1 = L·x and K2 = L·x² in GF(2^128), with L = AES(K, 0^128)."""

    def dbl_gf2to128(n):
        """Multiply by x: shift left, and XOR 0x87 if a bit fell off the top."""
        dbl_n = n << 1
        mask = (1 << 128) - 1
        flag = (dbl_n & (1 << 128)) > 0
        dbl_n &= mask
        if flag:
            dbl_n ^= 0x87
        return dbl_n

    base = aes_encrypt_block(b"\x00" * 16, key)
    basei = int.from_bytes(base, byteorder="big")
    k1 = dbl_gf2to128(basei)
    k2 = dbl_gf2to128(k1)

    return (k1.to_bytes(16, byteorder="big"), k2.to_bytes(16, byteorder="big"))


def cmac(key: bytes, message: bytes) -> bytes:
    """Return the 16-byte AES-CMAC tag of a message of any length, including empty."""
    k1, k2 = cmac_subkeys(key)

    n = len(message)
    step = max(0, ((n + 15) // 16) - 1)
    last = bytes(message[step * 16 :])
    if len(last) < 16:
        last = last + b"\x80" + b"\x00" * (16 - len(last) - 1)
        last = xor_bytes(last, k2)
    else:
        last = xor_bytes(last, k1)

    new_msg = bytes(message[: step * 16]) + last
    return cbc_mac(key, new_msg)
