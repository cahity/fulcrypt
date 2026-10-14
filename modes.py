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

from aes import encrypt_block, decrypt_block


BLOCK_SIZE = 16


def xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise ValueError(f"Byte sizes to be XOR'd should be same, you gave {len(a)} and {len(b)}.")

    return bytes(x^y for x, y in zip(a, b))


def split_blocks(data: bytes, block_size: int = BLOCK_SIZE) -> list[bytes]:
    if block_size < 1:
        raise ValueError(f"Block size ({block_size}) should be positive.")
    block_cnt, rem = divmod(len(data), block_size)
    if rem != 0:
        raise ValueError(f"Data length ({len(data)}) should be multiple of block size ({block_size}).")

    return [bytes(data[i*block_size:(i+1)*block_size]) for i in range(block_cnt)]


def pad(data: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    padding = block_size - (len(data) % block_size)
    return bytes(data) + bytes([padding] * padding)


def unpad(data: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    block_cnt, rem = divmod(len(data), block_size)

    if (
        block_cnt == 0
        or rem != 0
        or not (0 < data[-1] <= block_size)
        or data[-data[-1]:] != bytes([data[-1]]) * data[-1]
    ):
        raise ValueError("Invalid padding.")

    return bytes(data[:-data[-1]])


def ecb_encrypt(plaintext: bytes, key: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    if len(plaintext) % block_size != 0:
        raise ValueError("Plaintext should be block-aligned. Consider padding.")

    blocks = split_blocks(plaintext, block_size)
    return b"".join(encrypt_block(block, key) for block in blocks)


def ecb_decrypt(ciphertext: bytes, key: bytes, block_size: int = BLOCK_SIZE) -> bytes:
    if len(ciphertext) % block_size != 0:
        raise ValueError("Ciphertext should be block-aligned.")

    blocks = split_blocks(ciphertext, block_size)
    return b"".join(decrypt_block(block, key) for block in blocks)


def repeated_blocks(data: bytes, block_size: int = BLOCK_SIZE) -> int:
    blocks = split_blocks(data, block_size)
    return len(blocks) - len(set(blocks))
