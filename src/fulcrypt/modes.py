"""Block cipher modes of operation for AES (NIST SP 800-38A) and PKCS#7 padding (RFC 5652).

Design decisions:
- Plain bytes throughout. Modes only split data into blocks and XOR whole blocks;
  the GF(2^8) arithmetic stays inside aes.py.
- Padding is a separate layer. ECB and CBC take block-aligned data and reject anything
  else, as SP 800-38A defines them; callers pad explicitly: cbc_encrypt(pad(m), key, iv).
  This keeps the NIST test vectors (which have no padding) usable directly.
- CTR and OFB are stream modes: they XOR the data with a keystream, so they take data of
  any length and need no padding. Encryption and decryption are the same operation.
- The IV / counter block is passed in explicitly, so fixed test vectors work.
  encrypt_message() generates a random IV and prepends it to the ciphertext.
"""

import os

from fulcrypt.aes import decrypt_block, encrypt_block

BLOCK_SIZE = 16


def xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two byte strings of equal length."""
    if len(a) != len(b):
        raise ValueError(f"Byte strings must have equal length, got {len(a)} and {len(b)}.")

    return bytes(x ^ y for x, y in zip(a, b, strict=True))


def split_blocks(data: bytes, block_size: int = BLOCK_SIZE) -> list[bytes]:
    """Split block-aligned data into blocks."""
    if block_size < 1:
        raise ValueError(f"Block size must be positive, got {block_size}.")
    block_cnt, rem = divmod(len(data), block_size)
    if rem != 0:
        raise ValueError(
            f"Data length ({len(data)}) must be a multiple of the block size ({block_size})."
        )

    return [bytes(data[i * block_size : (i + 1) * block_size]) for i in range(block_cnt)]


def split_blocks_unaligned(data: bytes, block_size: int = BLOCK_SIZE) -> list[bytes]:
    """Split data of any length into blocks; the last block may be shorter."""
    if block_size < 1:
        raise ValueError(f"Block size must be positive, got {block_size}.")
    block_cnt, rem = divmod(len(data), block_size)

    if rem != 0:
        block_cnt += 1

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
        raise ValueError(f"IV must be exactly {BLOCK_SIZE} bytes, got {len(iv)}.")

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
        raise ValueError(f"IV must be exactly {BLOCK_SIZE} bytes, got {len(iv)}.")

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
        raise ValueError(
            f"Data must be an IV plus at least one {BLOCK_SIZE}-byte block, got {len(data)} bytes."
        )

    iv = data[:BLOCK_SIZE]
    ciphertext = data[BLOCK_SIZE:]
    msg_pad = cbc_decrypt(ciphertext, key, iv)
    msg = unpad(msg_pad, BLOCK_SIZE)

    return msg


def inc_counter(block: bytes, bits: int = 128) -> bytes:
    """Increment the low `bits` bits of a counter block mod 2^bits; the rest stays unchanged."""
    if len(block) != BLOCK_SIZE:
        raise ValueError(f"Counter block must be exactly {BLOCK_SIZE} bytes, got {len(block)}.")
    if not (1 <= bits <= 8 * BLOCK_SIZE):
        raise ValueError(f"Counter bits must be between 1 and {8 * BLOCK_SIZE}, got {bits}.")

    qbit, rbit = divmod(bits, 8)
    byte_len = qbit + 1
    new_block = bytearray(1) + bytearray(block)

    inc_int = int.from_bytes(new_block[-byte_len:], byteorder="big") + 1
    inc_int_mod = inc_int & ((1 << bits) - 1)
    inc_bytes = inc_int_mod.to_bytes(byte_len, byteorder="big")

    new_block[len(new_block) - byte_len + 1 :] = inc_bytes[1:]
    low = (1 << rbit) - 1
    new_block[-byte_len] = (new_block[-byte_len] & ~low) | (inc_bytes[0] & low)

    return bytes(new_block[1:])


def ctr_encrypt(data: bytes, key: bytes, counter_block: bytes, counter_bits: int = 128) -> bytes:
    """Encrypt data of any length in CTR mode, starting from the given counter block."""
    if len(counter_block) != BLOCK_SIZE:
        raise ValueError(
            f"Counter block must be exactly {BLOCK_SIZE} bytes, got {len(counter_block)}."
        )

    blocks = split_blocks_unaligned(data, BLOCK_SIZE)

    iv_ctr = counter_block
    ciphertext = []
    for block in blocks:
        ciphertext.append(xor_bytes(block, encrypt_block(iv_ctr, key)[: len(block)]))
        iv_ctr = inc_counter(iv_ctr, counter_bits)

    return b"".join(ciphertext)


def ctr_decrypt(data: bytes, key: bytes, counter_block: bytes, counter_bits: int = 128) -> bytes:
    """Decrypt data in CTR mode; the same operation as ctr_encrypt."""
    return ctr_encrypt(data, key, counter_block, counter_bits)


def ofb_encrypt(plaintext: bytes, key: bytes, iv: bytes) -> bytes:
    """Encrypt data of any length in OFB mode with the given IV."""
    if len(iv) != BLOCK_SIZE:
        raise ValueError(f"IV must be exactly {BLOCK_SIZE} bytes, got {len(iv)}.")

    blocks = split_blocks_unaligned(plaintext, BLOCK_SIZE)

    out = iv
    ciphertext = []
    for block in blocks:
        out = encrypt_block(out, key)
        ciphertext.append(xor_bytes(block, out[: len(block)]))

    return b"".join(ciphertext)


def ofb_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    """Decrypt data in OFB mode; the same operation as ofb_encrypt."""
    return ofb_encrypt(ciphertext, key, iv)
