"""Authenticated encryption: Encrypt-then-MAC (AES-CTR + HMAC-SHA256) and AES-GCM (SP 800-38D).

Design decisions:
- One master key; HKDF derives independent encryption and MAC keys (different info strings).
  Using one key for both AES and HMAC would void the security argument.
- Layout: nonce (16) || ciphertext || tag (32). The tag covers nonce || ciphertext, so the
  nonce can't be swapped either.
- Decryption verifies the tag first (in constant time) and decrypts only valid data.
- GCM's field is GF(2^128) with x^128 + x^7 + x^2 + x + 1, the usual (unreflected) modulus.
  Only blocks are reflected: GCM puts the coefficient of x^0 in the first bit of the block,
  so block_to_gcm / gcm_to_block reverse all 128 bits. The field itself stays standard.
"""

import os
from typing import Any

from fulcrypt.aes import AES_BLOCK_SIZE
from fulcrypt.aes import encrypt_block as aes_encrypt_block
from fulcrypt.elements import GF2n
from fulcrypt.hashes import SHA256_DIGEST_SIZE
from fulcrypt.kdf import hkdf
from fulcrypt.mac import constant_time_equal, hmac_sha256, hmac_sha256_verify
from fulcrypt.modes import ctr_decrypt, ctr_encrypt, inc_counter, xor_bytes

ETM_ENC_KEY_SIZE = 16  # AES-128
ETM_MAC_KEY_SIZE = SHA256_DIGEST_SIZE
ETM_NONCE_SIZE = AES_BLOCK_SIZE  # the CTR initial counter block
ETM_TAG_SIZE = SHA256_DIGEST_SIZE

# The GCM spec fixes this polynomial; everything below is read from it.
GCM_FIELD = GF2n((1 << 128) | (1 << 7) | (1 << 2) | (1 << 1) | (1 << 0))
GCM_BITS = GCM_FIELD.modulus.degree  # 128
GCM_BLOCK_SIZE = GCM_BITS // 8  # 16 bytes
# GCM is defined only for 128-bit block ciphers: a field element is exactly one AES block.
assert GCM_BLOCK_SIZE == AES_BLOCK_SIZE
GCM_COUNTER_BITS = 32  # inc32: only the last 32 bits of the counter block count
GCM_IV_SIZE = GCM_BLOCK_SIZE - GCM_COUNTER_BITS // 8  # 96-bit IV, the only size supported here


def etm_derive_keys(key: bytes) -> tuple[bytes, bytes]:
    """Derive (k_enc, k_mac) from a master key of any length: 16 and 32 bytes, via HKDF."""
    salt = b""
    info_enc = b"encryption"
    info_aut = b"authentication"

    k_enc = hkdf(key, salt, info_enc, ETM_ENC_KEY_SIZE)
    k_aut = hkdf(key, salt, info_aut, ETM_MAC_KEY_SIZE)

    return (k_enc, k_aut)


def etm_encrypt(key: bytes, message: bytes) -> bytes:
    """Encrypt with AES-CTR under a random nonce, then MAC; return nonce || ciphertext || tag."""
    k_enc, k_aut = etm_derive_keys(key)

    nonce = os.urandom(ETM_NONCE_SIZE)
    ciphertext = ctr_encrypt(message, k_enc, nonce)
    data = nonce + ciphertext
    tag = hmac_sha256(k_aut, data)

    return data + tag


def etm_decrypt(key: bytes, data: bytes) -> bytes:
    """Verify the tag, then decrypt; ValueError if the data is too short or was modified."""
    min_size = ETM_NONCE_SIZE + ETM_TAG_SIZE
    if len(data) < min_size:
        raise ValueError(
            f"Data must be at least {min_size} bytes (nonce and tag), got {len(data)}."
        )

    k_enc, k_aut = etm_derive_keys(key)
    data, tag = data[:-ETM_TAG_SIZE], data[-ETM_TAG_SIZE:]

    if not hmac_sha256_verify(k_aut, data, tag):
        raise ValueError("Authentication failed: wrong key, or the data was modified.")

    nonce = data[:ETM_NONCE_SIZE]
    ciphertext = data[ETM_NONCE_SIZE:]

    return ctr_decrypt(ciphertext, k_enc, nonce)


def block_to_gcm(block: bytes) -> Any:
    """Read a 16-byte block as a GCM_FIELD element: the first bit is the coefficient of x^0."""
    if len(block) != GCM_BLOCK_SIZE:
        raise ValueError(f"Block must be exactly {GCM_BLOCK_SIZE} bytes, got {len(block)}.")

    bits = int.from_bytes(block, "big")
    rev = 0
    for _ in range(GCM_BITS):
        rev = (rev << 1) | (bits & 1)
        bits >>= 1

    return GCM_FIELD(rev)


def gcm_to_block(element: Any) -> bytes:
    """Write a GCM_FIELD element as a 16-byte block, x^0 in the first bit."""
    bits = element.bits
    rev = 0
    for _ in range(GCM_BITS):
        rev = (rev << 1) | (bits & 1)
        bits >>= 1

    block = rev.to_bytes(GCM_BLOCK_SIZE, "big")
    return block


def ghash(h: bytes, aad: bytes, ciphertext: bytes) -> bytes:
    """GHASH: fold the padded AAD, padded ciphertext and a lengths block into Y = (Y + X) * H."""
    hgcm = block_to_gcm(h)

    # len(AAD) || len(C) in bits, each filling half a block (64 bits).
    half = GCM_BLOCK_SIZE // 2
    lens = (len(aad) * 8).to_bytes(half, "big") + (len(ciphertext) * 8).to_bytes(half, "big")

    aad_r = len(aad) % GCM_BLOCK_SIZE
    if aad_r != 0:
        aad = aad + b"\x00" * (GCM_BLOCK_SIZE - aad_r)

    cip_r = len(ciphertext) % GCM_BLOCK_SIZE
    if cip_r != 0:
        ciphertext = ciphertext + b"\x00" * (GCM_BLOCK_SIZE - cip_r)

    tot = aad + ciphertext + lens

    prev = b"\x00" * GCM_BLOCK_SIZE
    for i in range(0, len(tot), GCM_BLOCK_SIZE):
        curr = tot[i : i + GCM_BLOCK_SIZE]
        curr = xor_bytes(prev, curr)
        cgcm = hgcm * block_to_gcm(curr)
        prev = gcm_to_block(cgcm)

    return prev


def _gcm_setup(key: bytes, iv: bytes) -> tuple[bytes, bytes]:
    """Check the 96-bit IV; return (J0 = IV || 0^31 || 1, H = AES(K, 0^128))."""
    if len(iv) != GCM_IV_SIZE:
        raise ValueError(f"IV must be exactly {GCM_IV_SIZE} bytes, got {len(iv)}.")

    j0 = bytes(iv) + (1).to_bytes(GCM_COUNTER_BITS // 8, "big")
    h = aes_encrypt_block(b"\x00" * AES_BLOCK_SIZE, key)

    return (j0, h)


def gcm_encrypt(key: bytes, iv: bytes, plaintext: bytes, aad: bytes = b"") -> tuple[bytes, bytes]:
    """AES-GCM encrypt with a 96-bit IV; return (ciphertext, tag). AAD is authenticated only."""
    j0, h = _gcm_setup(key, iv)

    ciphertext = ctr_encrypt(plaintext, key, inc_counter(j0, GCM_COUNTER_BITS), GCM_COUNTER_BITS)
    tag = xor_bytes(aes_encrypt_block(j0, key), ghash(h, aad, ciphertext))

    return (ciphertext, tag)


def gcm_decrypt(key: bytes, iv: bytes, ciphertext: bytes, tag: bytes, aad: bytes = b"") -> bytes:
    """Check the tag in constant time, then decrypt; ValueError if authentication fails."""
    j0, h = _gcm_setup(key, iv)

    ntag = xor_bytes(aes_encrypt_block(j0, key), ghash(h, aad, ciphertext))

    if not constant_time_equal(tag, ntag):
        raise ValueError("Authentication failed: wrong key, or the data was modified.")

    plaintext = ctr_decrypt(ciphertext, key, inc_counter(j0, GCM_COUNTER_BITS), GCM_COUNTER_BITS)
    return plaintext
