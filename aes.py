"""AES block cipher (FIPS-197).

Design decisions:
- Bytes inside the cipher are GF(2^8) field elements (AES = GF2n(0x11B)), so the
  transformations read like the spec: XOR is +, the spec's "•" is *.
- The state is a 4x4 list indexed state[r][c], as in the spec. Input bytes fill it
  column by column: in[r + 4c] -> state[r][c]. Each column is one 32-bit word (section 3.4).
- Every transformation returns a new state and leaves its input unchanged (easier to debug and test).
"""

from elements import GF2n


Nb = 4  # Words per block
AES = GF2n(0x11B)


def bytes_to_state(block):
    if len(block) != 16:
        raise ValueError("Block must be 16 bytes long")

    state = [[0] * 4 for _ in range(4)]
    for c in range(4):
        for r in range(4):
            state[r][c] = AES(block[r + (4 * c)])
    return state

def state_to_bytes(state):
    block = bytearray(16)
    for c in range(4):
        for r in range(4):
            block[r + (4 * c)] = state[r][c].bits
    return bytes(block)

def compute_subbytes():
    """Compute the AES S-box and its inverse."""
    sbox = [0] * 256
    inv_sbox = [0] * 256

    for x in range(256):
        if x == 0:
            inv = 0
        else:
            inv = AES(x).inverse().bits

        y = 0
        for i in range(8):
            bit = (inv >> i) & 1
            bit ^= (inv >> ((i + 4) % 8)) & 1
            bit ^= (inv >> ((i + 5) % 8)) & 1
            bit ^= (inv >> ((i + 6) % 8)) & 1
            bit ^= (inv >> ((i + 7) % 8)) & 1
            y |= bit << i
        y ^= 0x63

        sbox[x] = y
        inv_sbox[y] = x

    return sbox, inv_sbox

sbox, inv_sbox = compute_subbytes()

def sub_bytes(state):
    return [[AES(sbox[byte.bits]) for byte in row] for row in state]

def inv_sub_bytes(state):
    return [[AES(inv_sbox[byte.bits]) for byte in row] for row in state]

def shift_rows(state):
    new_state = [[0] * 4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            new_state[r][c] = state[r][(c + r) % 4]
    return new_state

def inv_shift_rows(state):
    new_state = [[0] * 4 for _ in range(4)]
    for r in range(4):
        for c in range(4):
            new_state[r][c] = state[r][(c - r) % 4]
    return new_state

def mix_columns(state):
    new_state = [[0] * 4 for _ in range(4)]
    for c in range(4):
        for i in range(4):
            new_state[i][c] = (
                (AES(2) * state[i][c])
                + (AES(3) * state[(i+1) % 4][c])
                + state[(i+2) % 4][c]
                + state[(i + 3) % 4][c]
            )
        
    return new_state

def inv_mix_columns(state):
    new_state = [[0] * 4 for _ in range(4)]
    for c in range(4):
        for i in range(4):
            new_state[i][c] = (
                (AES(0x0e) * state[i][c])
                + (AES(0x0b) * state[(i+1) % 4][c])
                + (AES(0x0d) * state[(i+2) % 4][c])
                + (AES(0x09) * state[(i + 3) % 4][c])
            )

    return new_state

def add_round_key(state, round_key):
    new_state = [[0] * 4 for _ in range(4)]
    for i, w in enumerate(round_key):
        for j in range(4):
            new_state[j][i] = state[j][i] + w[j]
    return new_state

def rot_word(word):
    return [word[(i + 1) % 4] for i in range(4)]

def sub_word(word):
    return [AES(sbox[w.bits]) for w in word]

def rcon(i: int):
    if i < 1:
        raise ValueError(f"Rcon index must be at least 1, got {i} (Rcon[i]=x^(i-1) starts at i=1)")
    return [AES(0x02) ** (i-1), AES(0), AES(0), AES(0)]

def key_expansion(key: bytes):
    if len(key) not in [16, 24, 32]:
        raise ValueError(
            f"Only 16, 24, and 32 byte keys are accepted in AES. Provided key is {len(key)} bytes."
        )

    Nk = len(key) // 4
    Nr = 6 + Nk

    words = []
    for i in range(Nk):
        words.append([
            AES(key[4*i+0]),
            AES(key[4*i+1]),
            AES(key[4*i+2]),
            AES(key[4*i+3]),
        ])

    for i in range(Nk, Nb*(Nr+1)):
        tmp = words[-1]
        if i % Nk == 0:
            tmp = [w1 + w2 for w1, w2 in zip(sub_word(rot_word(tmp)), rcon(i//Nk))]
        elif Nk > 6 and i % Nk == 4:
            tmp = sub_word(tmp)

        words.append([w1 + w2 for w1, w2 in zip(words[i-Nk], tmp)])

    return words

def encrypt_block(block: bytes, key: bytes):
    if len(block) != Nb * 4:
        raise ValueError(
            f"Block size to encrypt should be {Nb * 4} bytes but it is {len(block)} bytes"
        )
    if len(key) not in [16, 24, 32]:
        raise ValueError(
            f"Only 16, 24, and 32 byte keys are accepted in AES. Provided key is {len(key)} bytes."
        )

    Nk = len(key) // 4
    Nr = 6 + Nk
    keywords = key_expansion(key)
    state = bytes_to_state(block)

    state = add_round_key(state, keywords[:4])

    for r in range(Nr-1):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, keywords[(r+1)*4:(r+2)*4])

    state = sub_bytes(state)
    state = shift_rows(state)
    # No mix_columns
    state = add_round_key(state, keywords[-4:])

    enc_block = state_to_bytes(state)
    return enc_block

def decrypt_block(block: bytes, key: bytes):
    if len(block) != Nb * 4:
        raise ValueError(
            f"Block size to decrypt should be {Nb * 4} bytes but it is {len(block)} bytes"
        )
    if len(key) not in [16, 24, 32]:
        raise ValueError(
            f"Only 16, 24, and 32 byte keys are accepted in AES. Provided key is {len(key)} bytes."
        )

    Nk = len(key) // 4
    Nr = 6 + Nk
    keywords = key_expansion(key)
    state = bytes_to_state(block)

    state = add_round_key(state, keywords[-4:])

    for r in range(Nr-2, -1, -1):
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)
        state = add_round_key(state, keywords[(r+1)*4:(r+2)*4])
        state = inv_mix_columns(state)

    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)
    state = add_round_key(state, keywords[:4])

    dec_block = state_to_bytes(state)
    return dec_block
