"""Key derivation: HKDF-SHA256 (RFC 5869).

Design decisions:
- Built on fulcrypt.mac.hmac_sha256, so it runs on our own HMAC and SHA-256.
- Extract and expand are public on their own: RFC 5869 publishes the PRK, so each
  half can be tested separately, and expand alone derives many keys from one PRK.
"""

from fulcrypt.hashes import SHA256_DIGEST_SIZE
from fulcrypt.mac import hmac_sha256


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    """Return the 32-byte PRK = HMAC(salt, ikm); an empty salt means 32 zero bytes."""
    if len(salt) == 0:
        salt = b"\x00" * SHA256_DIGEST_SIZE
    return hmac_sha256(salt, ikm)


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """Stretch a PRK to `length` bytes: T(i) = HMAC(prk, T(i-1) || info || i), i = 1, 2, ..."""
    if not (0 <= length <= 255 * SHA256_DIGEST_SIZE):
        raise ValueError(
            f"Length must be between 0 and {255 * SHA256_DIGEST_SIZE} bytes, got {length}."
        )
    num_blocks = (length + SHA256_DIGEST_SIZE - 1) // SHA256_DIGEST_SIZE
    blocks = []
    curr_block = b""
    for block_i in range(num_blocks):
        curr_block = hmac_sha256(
            prk, curr_block + info + (block_i + 1).to_bytes(1, byteorder="big")
        )
        blocks.append(curr_block)

    return b"".join(blocks)[:length]


def hkdf(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """Derive `length` bytes from ikm: extract with salt, then expand with info."""
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)
