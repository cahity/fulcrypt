"""Block cipher modes of operation for AES (NIST SP 800-38A) and PKCS#7 padding (RFC 5652).

Design decisions:
- Plain bytes throughout. Modes only split data into blocks and XOR whole blocks;
  the GF(2^8) arithmetic stays inside aes.py.
- Padding is a separate layer. ECB and CBC take block-aligned data and reject anything
  else, as SP 800-38A defines them; callers pad explicitly: cbc_encrypt(pad(m), key, iv).
  This keeps the NIST test vectors (which have no padding) usable directly.
- The IV is passed in explicitly, so fixed test vectors work. encrypt_message() generates
  a random IV and prepends it to the ciphertext.
"""

import os

from aes import decrypt_block, encrypt_block

BLOCK_SIZE = 16


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings of equal length."""
    if len(a) != len(b):
        raise ValueError(f"Byte sizes to be XOR'd should be same, you gave {len(a)} and {len(b)}.")

    return bytes(x ^ y for x, y in zip(a, b, strict=True))


def split_blocks(data: bytes, block_size: int = BLOCK_SIZE) -> list[bytes]:
    """Split block-aligned data into blocks."""
    if block_size < 1:
        raise ValueError(f"Block size ({block_size}) should be positive.")
    block_cnt, rem = divmod(len(data), block_size)
    if rem != 0:
        raise ValueError(
            f"Data length ({len(data)}) should be multiple of block size ({block_size})."
        )

    return [bytes(data[i * block_size : (i + 1) * block_size]) for i in range(block_cnt)]


def pad(data: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    """Add PKCS#7 padding (always at least one byte)."""
    padding = block_size - (len(data) % block_size)
    return bytes(data) + bytes([padding] * padding)


def unpad(data: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    """Remove and validate PKCS#7 padding; ValueError if invalid."""
    block_cnt, rem = divmod(len(data), block_size)

    if (
        block_cnt == 0
        or rem != 0
        or not (0 < data[-1] <= block_size)
        or data[-data[-1] :] != bytes([data[-1]]) * data[-1]
    ):
        raise ValueError("Invalid padding.")

    return bytes(data[: -data[-1]])


def ecb_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt block-aligned data in ECB mode."""
    blocks = split_blocks(plaintext, BLOCK_SIZE)
    return b"".join(encrypt_block(block, key) for block in blocks)


def ecb_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt block-aligned data in ECB mode."""
    blocks = split_blocks(ciphertext, BLOCK_SIZE)
    return b"".join(decrypt_block(block, key) for block in blocks)


def repeated_blocks(data: bytes, block_size: int = BLOCK_SIZE) -> int:
    """Count blocks that duplicate an earlier block."""
    blocks = split_blocks(data, block_size)
    return len(blocks) - len(set(blocks))


def cbc_encrypt(plaintext: bytes, key: bytes, iv: bytes) -> bytes:
    """Encrypt block-aligned data in CBC mode with the given IV."""
    if len(iv) != BLOCK_SIZE:
        raise ValueError(f"IV must be exactly {BLOCK_SIZE} bytes.")

    blocks = split_blocks(plaintext, BLOCK_SIZE)

    prev = iv
    ciphertext = []
    for block in blocks:
        ciphertext.append(encrypt_block(xor_bytes(block, prev), key))
        prev = ciphertext[-1]

    return b"".join(ciphertext)


def cbc_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    """Decrypt block-aligned data in CBC mode with the given IV."""
    if len(iv) != BLOCK_SIZE:
        raise ValueError(f"IV must be exactly {BLOCK_SIZE} bytes.")

    blocks = split_blocks(ciphertext, BLOCK_SIZE)
    blocks = [iv] + blocks

    plaintext = []
    # Pairs (previous, current): the lists differ in length by one on purpose.
    for prev, curr in zip(blocks, blocks[1:], strict=False):
        plaintext.append(xor_bytes(decrypt_block(curr, key), prev))

    return b"".join(plaintext)


def encrypt_message(message: bytes, key: bytes) -> bytes:
    """Pad and CBC-encrypt a message under a random IV; return IV || ciphertext."""
    iv = os.urandom(BLOCK_SIZE)
    msg_pad = pad(message, BLOCK_SIZE)
    return iv + cbc_encrypt(msg_pad, key, iv)


def decrypt_message(data: bytes, key: bytes) -> bytes:
    """Split off the IV, CBC-decrypt and unpad; inverse of encrypt_message."""
    if len(data) % BLOCK_SIZE != 0 or len(data) <= BLOCK_SIZE:
        raise ValueError(f"Invalid data size to decrypt at {len(data)} bytes.")

    iv = data[:BLOCK_SIZE]
    ciphertext = data[BLOCK_SIZE:]
    msg_pad = cbc_decrypt(ciphertext, key, iv)
    msg = unpad(msg_pad, BLOCK_SIZE)

    return msg
