"""Authenticated encryption: Encrypt-then-MAC (AES-CTR + HMAC-SHA256).

Design decisions:
- One master key; HKDF derives independent encryption and MAC keys (different info strings).
  Using one key for both AES and HMAC would void the security argument.
- Layout: nonce (16) || ciphertext || tag (32). The tag covers nonce || ciphertext, so the
  nonce can't be swapped either.
- Decryption verifies the tag first (in constant time) and decrypts only valid data.
"""

import os

from fulcrypt.kdf import hkdf
from fulcrypt.mac import hmac_sha256, hmac_sha256_verify
from fulcrypt.modes import ctr_decrypt, ctr_encrypt


def etm_derive_keys(key: bytes) -> tuple[bytes, bytes]:
    """Derive (k_enc, k_mac) from a master key of any length: 16 and 32 bytes, via HKDF."""
    salt = b""
    info_enc = b"encryption"
    info_aut = b"authentication"

    k_enc = hkdf(key, salt, info_enc, 16)
    k_aut = hkdf(key, salt, info_aut, 32)

    return (k_enc, k_aut)


def etm_encrypt(key: bytes, message: bytes) -> bytes:
    """Encrypt with AES-CTR under a random nonce, then MAC; return nonce || ciphertext || tag."""
    k_enc, k_aut = etm_derive_keys(key)

    nonce = os.urandom(16)
    ciphertext = ctr_encrypt(message, k_enc, nonce)
    data = nonce + ciphertext
    tag = hmac_sha256(k_aut, data)

    return data + tag


def etm_decrypt(key: bytes, data: bytes) -> bytes:
    """Verify the tag, then decrypt; ValueError if the data is too short or was modified."""
    if len(data) < 48:
        raise ValueError(f"Data must be at least 48 bytes (nonce and tag), got {len(data)}.")

    k_enc, k_aut = etm_derive_keys(key)
    data, tag = data[:-32], data[-32:]

    if not hmac_sha256_verify(k_aut, data, tag):
        raise ValueError("Authentication failed: wrong key, or the data was modified.")

    nonce = data[:16]
    ciphertext = data[16:]

    return ctr_decrypt(ciphertext, k_enc, nonce)
