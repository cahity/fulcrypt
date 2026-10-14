# fulcrypt

Cryptography primitives implemented from scratch, to learn how they work.

> [!WARNING]
> **For learning only. Do not use this to protect real data.**
> The code is not constant-time, so it leaks information through timing, and it has not been
> audited. For real applications, use a maintained library such as
> [`cryptography`](https://cryptography.io).

## What's inside

| Module | Contents |
|---|---|
| `fulcrypt.elements` | Element classes: `GF2Poly` (polynomials over GF(2)), `ZMod(n)` (integers mod n), `GF2n(modulus)` (finite fields GF(2ⁿ)) |
| `fulcrypt.algorithms` | Generic extended Euclid (`xgcd`), `inverse_mod`, `power` (square-and-multiply); work on `int` and every element class |
| `fulcrypt.algebra` | Brute-force checks of the group, ring and field axioms on small structures |
| `fulcrypt.aes` | AES (FIPS-197) with 128, 192 and 256-bit keys, built on GF(2⁸) field elements |
| `fulcrypt.modes` | PKCS#7 padding, ECB and CBC modes (NIST SP 800-38A) |

Every function is checked against the published test vectors of its standard (FIPS-197, NIST SP 800-38A).

## Usage

Arithmetic reads like the math:

```python
>>> from fulcrypt.elements import GF2n, ZMod
>>> Z7 = ZMod(7)
>>> Z7(3) * Z7(5)
1 (mod 7)
>>> Z7(3) ** -1
5 (mod 7)
>>> AES = GF2n(0x11B)           # GF(2^8) with the AES polynomial x^8 + x^4 + x^3 + x + 1
>>> AES(0x57) * AES(0x83)
0xc1
```

AES and CBC:

```python
>>> from fulcrypt import aes, modes
>>> key = bytes(range(16))
>>> aes.encrypt_block(bytes.fromhex("00112233445566778899aabbccddeeff"), key).hex()
'69c4e0d86a7b0430d8cdb78070b4c55a'
>>> data = modes.encrypt_message(b"attack at dawn", key)   # random IV || CBC ciphertext
>>> modes.decrypt_message(data, key)
b'attack at dawn'
```

More walk-throughs are in [`examples/example.py`](examples/example.py).

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repository-url>
cd fulcrypt
uv sync
git config core.hooksPath .githooks   # enable the pre-commit checks
```

## Development

All checks run before every commit (`.githooks/pre-commit`); they only report, never change files:

```bash
uv run ruff format --check .   # formatting (black-style)
uv run ruff check .            # PEP 8, imports, docstrings, naming
uv run mypy                    # types
uv run pytest -q               # tests
```

To fix formatting and lint automatically: `uv run ruff format . && uv run ruff check --fix .`

## How this was built

I wrote the implementation in `src/fulcrypt/` myself, as a way to learn the material.
I used Claude Code along the way: it planned the steps, wrote the tests and the docstrings,
and reviewed my code.

## License

[MIT](LICENSE) © Y. Cahit Yıldırım
